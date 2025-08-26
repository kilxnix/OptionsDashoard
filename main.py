from flask import Flask, jsonify, request
from datetime import datetime
import os
import json
import glob
import pandas as pd
import numpy as np
import requests
import time
import yfinance as yf
from run_autonomous_scan import run_autonomous_scan
from quantitative_analyzer import is_in_bollinger_squeeze, calculate_relative_volume

app = Flask(__name__)

# Alpha Vantage API integration - no rate limiting needed with subscription


def make_json_safe(obj):
    """Recursively convert pandas and numpy objects to JSON-serializable forms."""
    # Handle numpy arrays first to avoid truth value ambiguity
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Handle pandas NA and None values
    if obj is pd.NA or obj is None:
        return None

    # Check for pandas NA values safely (only for scalar values)
    try:
        if hasattr(obj, '__len__') and len(obj) > 1:
            # This is an array-like object, don't use pd.isna
            pass
        elif pd.isna(obj):
            return None
    except (ValueError, TypeError):
        # pd.isna failed, continue with other checks
        pass

    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                        np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    return obj


@app.route("/")
def index():
    return jsonify({
        "status": "online",
        "message": "Autonomous scanner is running.",
        "timestamp": datetime.now().isoformat()
    })


@app.route("/health")
def health():
    return "OK", 200


@app.route("/scan/parameters", methods=["GET"])
def get_scan_parameters():
    """Get available scan parameters and their default values"""
    return jsonify({
        "status":
        "success",
        "parameters": {
            "auto_refresh": {
                "type":
                "boolean",
                "default":
                True,
                "description":
                "Auto-refresh symbols from Alpha Vantage before scanning"
            },
            "limit": {
                "type": "integer",
                "default": 25,
                "description": "Maximum number of symbols to process"
            },
            "min_delta": {
                "type": "float",
                "default": 0.25,
                "description": "Minimum option delta (0.0 to 1.0)"
            },
            "max_delta": {
                "type": "float",
                "default": 0.68,
                "description": "Maximum option delta (0.0 to 1.0)"
            },
            "min_price": {
                "type": "float",
                "default": 0.01,
                "description": "Minimum option price in dollars"
            },
            "max_price": {
                "type": "float",
                "default": 0.10,
                "description": "Maximum option price in dollars"
            },
            "min_days": {
                "type": "integer",
                "default": 2,
                "description": "Minimum days to expiration"
            },
            "max_days": {
                "type": "integer",
                "default": 16,
                "description": "Maximum days to expiration"
            },
            "iv_percentile": {
                "type":
                "integer",
                "default":
                None,
                "description":
                "Minimum IV percentile threshold (0-100) - DISABLED by default"
            }
        },
        "example_usage":
        "/scan?min_delta=0.3&max_delta=0.7&min_price=0.05&max_price=0.20&min_days=1&max_days=30"
    })


@app.route("/scan", methods=["GET", "POST"])
def trigger_scan():
    # Handle both GET (query params) and POST (JSON body) requests
    if request.method == 'POST' and request.is_json:
        data = request.get_json()
        auto_refresh = data.get('auto_refresh', True)
        limit = int(data.get('limit', 0))  # 0 = unlimited
        min_delta = float(data.get('min_delta', 0.25))
        max_delta = float(data.get('max_delta', 0.68))
        min_price = float(data.get('min_price', 0.01))
        max_price = float(data.get('max_price', 0.10))
        min_days = int(data.get('min_days', 0))
        max_days = int(data.get('max_days', 30))
        iv_percentile = data.get('iv_percentile', None)
        if iv_percentile is not None:
            iv_percentile = int(iv_percentile)
    else:
        # GET request - use query parameters
        auto_refresh = request.args.get('auto_refresh',
                                        'true').lower() == 'true'
        limit = int(request.args.get('limit', 0))  # 0 = unlimited
        min_delta = float(request.args.get('min_delta', 0.25))
        max_delta = float(request.args.get('max_delta', 0.68))
        min_price = float(request.args.get('min_price', 0.01))
        max_price = float(request.args.get('max_price', 0.10))
        min_days = int(request.args.get('min_days', 2))
        max_days = int(request.args.get('max_days', 16))
        iv_percentile = request.args.get('iv_percentile')
        iv_percentile = int(iv_percentile) if iv_percentile else None

    result = run_autonomous_scan(dry_run=False,
                                 auto_refresh_symbols=auto_refresh,
                                 symbol_limit=limit,
                                 min_delta=min_delta,
                                 max_delta=max_delta,
                                 min_price=min_price,
                                 max_price=max_price,
                                 time_to_expiry_range=(min_days, max_days),
                                 iv_percentile_threshold=iv_percentile)

    if not result or not result.get("results"):
        return jsonify({
            "status":
            "no-results",
            "message":
            "Scanner ran but found no valid trade plans.",
            "digest":
            result["digest"] if result else "No output",
            "symbols_processed":
            result.get("symbols_processed", 0) if result else 0
        }), 200

    # Get summary stats
    results = result.get("results", {})
    top_scores = sorted([(k, v.get("confluence", {}).get("score", 0))
                         for k, v in results.items()],
                        key=lambda x: x[1],
                        reverse=True)

    return jsonify({
        "status":
        "completed",
        "digest":
        result["digest"],
        "symbols_processed":
        result.get("symbols_processed", 0),
        "opportunities_found":
        len(results),
        "top_3_symbols":
        [f"{sym} ({score:.1f})" for sym, score in top_scores[:3]],
        "scan_parameters": {
            "min_delta": min_delta,
            "max_delta": max_delta,
            "min_price": min_price,
            "max_price": max_price,
            "days_to_expiry": f"{min_days}-{max_days}",
            "iv_percentile_threshold": iv_percentile,
            "auto_refresh": auto_refresh,
            "symbol_limit": limit
        }
    }), 200


@app.route("/plans", methods=["GET"])
def get_sorted_plans():
    """Return sorted trading plans from the most recent scan"""
    try:
        # Get query parameters for sorting
        sort_by = request.args.get(
            'sort_by', 'confluence_score')  # Default sort by confluence score
        order = request.args.get('order', 'desc')  # Default descending order
        date = request.args.get(
            'date',
            datetime.now().strftime('%Y-%m-%d'))  # Default to today

        # Find the most recent progressive results file
        base_dir = './TradingPlans'
        json_pattern = f'progressive_results_{date}.json'
        json_file = os.path.join(base_dir, json_pattern)

        if not os.path.exists(json_file):
            # Try to find the most recent file if today's doesn't exist
            pattern = os.path.join(base_dir, 'progressive_results_*.json')
            files = glob.glob(pattern)
            if not files:
                return jsonify({
                    "status": "error",
                    "message": "No trading plans found"
                }), 404
            json_file = max(files, key=os.path.getctime)

        # Load the JSON data
        with open(json_file, 'r') as f:
            results = json.load(f)

        # Convert to sortable format
        plans = []
        for symbol, data in results.items():
            plan_data = {
                'symbol': symbol,
                'confluence_score': data.get('confluence', {}).get('score', 0),
                'bias': data.get('confluence', {}).get('bias', 'N/A'),
                'entry_price': 0,
                'target_price': 0,
                'option_type': 'N/A',
                'strike': 'N/A',
                'expiration': 'N/A',
                'position_size': 0,
                'max_hold_time': 'N/A'
            }

            # Extract trade plan details if available
            if 'trade_plan' in data and data['trade_plan']:
                tp = data['trade_plan']
                plan_data.update({
                    'entry_price':
                    tp.get('entry_price', 0),
                    'target_price':
                    tp.get('initial_target', 0),
                    'option_type':
                    tp.get('type', 'N/A'),
                    'strike':
                    tp.get('strike', 'N/A'),
                    'expiration':
                    tp.get('expiration', 'N/A'),
                    'position_size':
                    tp.get('position_size', 0),
                    'max_hold_time':
                    tp.get('max_hold_time', 'N/A')
                })

            plans.append(plan_data)

        # Sort the plans
        reverse_order = order.lower() == 'desc'

        if sort_by == 'confluence_score':
            plans.sort(key=lambda x: x['confluence_score'],
                       reverse=reverse_order)
        elif sort_by == 'entry_price':
            plans.sort(key=lambda x: x['entry_price'], reverse=reverse_order)
        elif sort_by == 'target_price':
            plans.sort(key=lambda x: x['target_price'], reverse=reverse_order)
        elif sort_by == 'symbol':
            plans.sort(key=lambda x: x['symbol'], reverse=reverse_order)
        else:
            plans.sort(key=lambda x: x['confluence_score'],
                       reverse=True)  # Default fallback

        return jsonify({
            "status": "success",
            "total_plans": len(plans),
            "sort_by": sort_by,
            "order": order,
            "date": date,
            "plans": plans
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error retrieving plans: {str(e)}"
        }), 500


@app.route("/plans/all", methods=["GET"])
def get_all_plans():
    """Return all trading plans from ALL progressive results files"""
    try:
        # Get query parameters for sorting
        sort_by = request.args.get('sort_by', 'confluence_score')
        order = request.args.get('order', 'desc')
        limit = request.args.get('limit', type=int)  # Optional limit

        base_dir = './TradingPlans'
        pattern = os.path.join(base_dir, 'progressive_results_*.json')
        files = glob.glob(pattern)

        if not files:
            return jsonify({
                "status": "error",
                "message": "No trading plans found"
            }), 404

        # Load and merge all files
        all_plans = []
        files_processed = []

        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    results = json.load(f)

                # Extract date from filename for tracking
                filename = os.path.basename(file_path)
                date_part = filename.replace('progressive_results_',
                                             '').replace('.json', '')
                files_processed.append(date_part)

                # Convert each symbol's data to plan format
                for symbol, data in results.items():
                    plan_data = {
                        'symbol':
                        symbol,
                        'scan_date':
                        date_part,
                        'confluence_score':
                        data.get('confluence', {}).get('score', 0),
                        'bias':
                        data.get('confluence', {}).get('bias', 'N/A'),
                        'entry_price':
                        0,
                        'target_price':
                        0,
                        'option_type':
                        'N/A',
                        'strike':
                        'N/A',
                        'expiration':
                        'N/A',
                        'expiration_date':
                        None,  # For proper date sorting
                        'position_size':
                        0,
                        'max_hold_time':
                        'N/A'
                    }

                    # Extract trade plan details if available
                    if 'trade_plan' in data and data['trade_plan']:
                        tp = data['trade_plan']
                        expiration_str = tp.get('expiration', 'N/A')
                        plan_data.update({
                            'entry_price':
                            tp.get('entry_price', 0),
                            'target_price':
                            tp.get('initial_target', 0),
                            'option_type':
                            tp.get('type', 'N/A'),
                            'strike':
                            tp.get('strike', 'N/A'),
                            'expiration':
                            expiration_str,
                            'position_size':
                            tp.get('position_size', 0),
                            'max_hold_time':
                            tp.get('max_hold_time', 'N/A')
                        })

                        # Convert expiration to datetime for sorting
                        try:
                            if expiration_str != 'N/A':
                                plan_data['expiration_date'] = pd.to_datetime(
                                    expiration_str)
                        except:
                            plan_data['expiration_date'] = None

                    all_plans.append(plan_data)

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue

        if not all_plans:
            return jsonify({
                "status": "error",
                "message": "No valid plans found in any files"
            }), 404

        # Sort the plans
        reverse_order = order.lower() == 'desc'

        if sort_by == 'confluence_score':
            all_plans.sort(key=lambda x: x['confluence_score'],
                           reverse=reverse_order)
        elif sort_by == 'entry_price':
            all_plans.sort(key=lambda x: x['entry_price'],
                           reverse=reverse_order)
        elif sort_by == 'target_price':
            all_plans.sort(key=lambda x: x['target_price'],
                           reverse=reverse_order)
        elif sort_by == 'symbol':
            all_plans.sort(key=lambda x: x['symbol'], reverse=reverse_order)
        elif sort_by == 'scan_date':
            all_plans.sort(key=lambda x: x['scan_date'], reverse=reverse_order)
        elif sort_by == 'expiration' or sort_by == 'expiration_date':
            # Sort by expiration date, putting None values at the end
            all_plans.sort(
                key=lambda x: x['expiration_date']
                if x['expiration_date'] is not None else pd.Timestamp.max,
                reverse=reverse_order)
        else:
            all_plans.sort(key=lambda x: x['confluence_score'], reverse=True)

        # Apply limit if specified
        if limit and limit > 0:
            all_plans = all_plans[:limit]

        return jsonify({
            "status": "success",
            "total_plans": len(all_plans),
            "files_processed": files_processed,
            "sort_by": sort_by,
            "order": order,
            "plans": all_plans
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error retrieving all plans: {str(e)}"
        }), 500


def fetch_alphavantage_top_symbols():
    """Fetch top gainers, losers, and most active symbols from Alpha Vantage"""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise Exception("ALPHA_VANTAGE_API_KEY environment variable not set")

    url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={api_key}'

    try:
        print(
            "Fetching top gainers, losers, and most active from Alpha Vantage..."
        )
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        if 'Error Message' in data:
            raise Exception(f"Alpha Vantage error: {data['Error Message']}")

        if 'Information' in data:
            raise Exception(f"Alpha Vantage info: {data['Information']}")

        all_symbols = []
        categories = ['top_gainers', 'top_losers', 'most_actively_traded']

        for category in categories:
            if category in data:
                category_symbols = [item['ticker'] for item in data[category]]
                all_symbols.extend(category_symbols)
                print(
                    f"Fetched {len(category_symbols)} symbols from {category}")

        # Add common optionable stocks as backup
        backup_symbols = [
            'SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
            'NVDA', 'META', 'AMD', 'INTC', 'NFLX', 'CRM', 'UBER', 'LYFT',
            'BABA', 'DIS', 'BA', 'GE', 'F', 'GM', 'T', 'VZ', 'JPM', 'BAC',
            'WFC', 'C', 'GS', 'MS', 'XOM', 'CVX', 'KO', 'PEP', 'WMT', 'TGT'
        ]

        # Add backup symbols if we have fewer than 40 unique symbols
        if len(all_symbols) < 40:
            for symbol in backup_symbols:
                if symbol not in all_symbols:
                    all_symbols.append(symbol)
                    if len(all_symbols) >= 60:  # Cap at reasonable number
                        break

        # Remove duplicates while preserving order
        unique_symbols = list(dict.fromkeys(all_symbols))
        print(
            f"Total unique symbols (including backup): {len(unique_symbols)}")

        return {
            'all_symbols':
            unique_symbols,
            'top_gainers':
            [item['ticker'] for item in data.get('top_gainers', [])],
            'top_losers':
            [item['ticker'] for item in data.get('top_losers', [])],
            'most_active':
            [item['ticker'] for item in data.get('most_actively_traded', [])]
        }

    except Exception as e:
        print(f"Error fetching Alpha Vantage data: {e}")
        # Return backup symbols as fallback
        backup_symbols = [
            'SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
            'NVDA', 'META', 'AMD', 'INTC', 'NFLX', 'UBER', 'DIS', 'F'
        ]
        return {
            'all_symbols': backup_symbols,
            'top_gainers': [],
            'top_losers': [],
            'most_active': []
        }


@app.route("/screener/symbols", methods=["GET"])
def get_screener_symbols():
    """Get symbols from Alpha Vantage screeners"""
    try:
        # Get category type from query params
        category = request.args.get('category', 'all')
        valid_categories = ['all', 'top_gainers', 'top_losers', 'most_active']

        if category not in valid_categories:
            return jsonify({
                "status":
                "error",
                "message":
                f"Invalid category. Valid options: {valid_categories}"
            }), 400

        data = fetch_alphavantage_top_symbols()

        if category == 'all':
            symbols = data['all_symbols']
        elif category == 'top_gainers':
            symbols = data['top_gainers']
        elif category == 'top_losers':
            symbols = data['top_losers']
        elif category == 'most_active':
            symbols = data['most_active']

        return jsonify({
            "status": "success",
            "category": category,
            "total_symbols": len(symbols),
            "symbols": symbols
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error fetching symbols: {str(e)}"
        }), 500


@app.route("/screener/all", methods=["GET"])
def get_all_screener_symbols():
    """Get symbols from Alpha Vantage top gainers, losers, and most active"""
    try:
        data = fetch_alphavantage_top_symbols()

        response_data = {
            "status": "success",
            "source": "alpha_vantage",
            "categories": {
                "top_gainers": data['top_gainers'],
                "top_losers": data['top_losers'],
                "most_active": data['most_active']
            },
            "combined_unique_symbols": data['all_symbols'],
            "total_unique": len(data['all_symbols']),
            "breakdown": {
                "top_gainers": len(data['top_gainers']),
                "top_losers": len(data['top_losers']),
                "most_active": len(data['most_active'])
            }
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error fetching all symbols: {str(e)}"
        }), 500


@app.route("/screener/fallback", methods=["GET"])
def get_fallback_symbols():
    """Get symbols from database when Yahoo Finance is rate limited"""
    try:
        from db_client import fetch_tickers_from_db

        db_symbols = fetch_tickers_from_db()

        return jsonify({
            "status": "success",
            "source": "database",
            "total_symbols": len(db_symbols),
            "symbols": db_symbols,
            "message": "Using database symbols as fallback"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error fetching fallback symbols: {str(e)}"
        }), 500


@app.route("/screener/update-db", methods=["POST", "GET"])
def update_database_symbols():
    """Update database with fresh symbols from Yahoo Finance screeners"""
    try:
        # Handle both GET and POST requests
        if request.method == 'GET':
            # For GET requests, use query parameters
            mode = request.args.get('mode', 'replace')
            custom_symbols_str = request.args.get('symbols', '')
            custom_symbols = custom_symbols_str.split(
                ',') if custom_symbols_str else []
        else:
            # For POST requests, use JSON body
            data = request.get_json() if request.is_json else {}
            mode = data.get('mode', 'replace')  # Default to replace
            custom_symbols = data.get('symbols',
                                      [])  # Allow custom symbol list

        if custom_symbols:
            # Use provided symbols
            symbols_to_add = custom_symbols
            source = "custom"
        else:
            # Fetch from Alpha Vantage
            data = fetch_alphavantage_top_symbols()
            symbols_to_add = data['all_symbols']
            source = "alpha_vantage"

        if not symbols_to_add:
            return jsonify({
                "status": "error",
                "message": "No symbols to add to database"
            }), 400

        # Update database
        from db_client import update_tickers_in_db
        result = update_tickers_in_db(symbols_to_add, mode=mode)

        return jsonify({
            "status":
            "success",
            "source":
            source,
            "mode":
            mode,
            "symbols_added":
            len(symbols_to_add),
            "total_in_database":
            result["total_in_db"],
            "message":
            f"Database updated successfully with {len(symbols_to_add)} symbols"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error updating database: {str(e)}"
        }), 500


@app.route("/performance/update", methods=["POST", "GET"])
def update_performance():
    """Update performance tracking for all tracked options"""
    try:
        from performance_tracker import update_performance_tracking

        updated_count = update_performance_tracking()

        return jsonify({
            "status": "success",
            "message": f"Updated performance for {updated_count} options",
            "updated_count": updated_count,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error updating performance: {str(e)}"
        }), 500


@app.route("/performance/analyze", methods=["GET"])
def analyze_performance():
    """Get comprehensive performance analysis"""
    try:
        from performance_tracker import analyze_performance

        metrics, suggestions = analyze_performance()

        return jsonify({
            "status": "success",
            "performance_metrics": metrics,
            "improvement_suggestions": suggestions,
            "analysis_timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error analyzing performance: {str(e)}"
        }), 500


@app.route("/performance/report", methods=["GET"])
def get_performance_report():
    """Get human-readable performance report"""
    try:
        from performance_tracker import PerformanceTracker

        tracker = PerformanceTracker()
        metrics = tracker.calculate_performance_metrics()
        suggestions = tracker.get_improvement_suggestions()

        # Format as readable text
        report = f"""
PERFORMANCE REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

OVERALL STATISTICS:
• Total Predictions: {metrics.get('total_predictions', 0)}
• Win Rate: {metrics.get('win_rate', 0):.1f}%
• Targets Hit: {metrics.get('targets_hit', 0)}
• Stops Hit: {metrics.get('stops_hit', 0)}
• Still Active: {metrics.get('still_active', 0)}

CONFLUENCE SCORE ACCURACY:
"""

        for score_range, data in metrics.get('confluence_score_accuracy',
                                             {}).items():
            report += f"• {score_range}: {data['win_rate']:.1f}% ({data['wins']}/{data['total']})\n"

        report += "\nBIAS ACCURACY:\n"
        for bias, data in metrics.get('bias_accuracy', {}).items():
            report += f"• {bias}: {data['win_rate']:.1f}% ({data['wins']}/{data['total']})\n"

        if metrics.get('best_performers'):
            report += "\nBEST PERFORMERS:\n"
            for performer in metrics['best_performers'][:5]:
                report += f"• {performer['symbol']}: +{performer['max_profit']:.1f}% (Score: {performer['confluence_score']:.1f})\n"

        if metrics.get('worst_performers'):
            report += "\nWORST PERFORMERS:\n"
            for performer in metrics['worst_performers'][:5]:
                report += f"• {performer['symbol']}: {performer['max_profit']:.1f}% (Score: {performer['confluence_score']:.1f})\n"

        report += "\nIMPROVEMENT SUGGESTIONS:\n"
        for suggestion in suggestions:
            report += f"• {suggestion}\n"

        return report, 200, {'Content-Type': 'text/plain; charset=utf-8'}

    except Exception as e:
        return f"Error generating performance report: {str(e)}", 500


@app.route("/enhanced-scan", methods=["GET", "POST"])
def enhanced_scan():
    """Enhanced options scanning with complex analysis"""
    try:
        # Get scan parameters
        if request.method == 'POST' and request.is_json:
            data = request.get_json()
            symbols = data.get('symbols', [])
            sector_filter = data.get('sector', None)
            min_score = data.get('min_score', 65)
            scan_date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
            max_symbols = data.get('max_symbols', 400)
        else:
            symbols = request.args.getlist('symbols')
            sector_filter = request.args.get('sector', None)
            min_score = float(request.args.get('min_score', 65))
            scan_date = request.args.get('date',
                                         datetime.now().strftime('%Y-%m-%d'))
            max_symbols = int(request.args.get('max_symbols', 400))

        # Use proper auto-discovery like explosive scan
        if not symbols:
            # Get symbols from Alpha Vantage screeners (same as explosive scan)
            try:
                data = fetch_alphavantage_top_symbols()
                symbols = data['all_symbols']
                print(
                    f"🔍 Auto-discovered {len(symbols)} symbols from Alpha Vantage screeners"
                )
            except Exception as e:
                print(f"⚠️ Auto-discovery failed, using fallback: {e}")
                # Fallback to scanner_core method
                from scanner_core import get_optionable_stocks_with_volume
                symbols = get_optionable_stocks_with_volume()

            if max_symbols and len(symbols) > max_symbols:
                symbols = symbols[:max_symbols]
                print(f"🎯 Limited to {max_symbols} symbols for enhanced scan")

        # Run enhanced scan
        scanner = EnhancedOptionsScanner(os.getenv('ALPHA_VANTAGE_API_KEY'))
        results = scanner.run_comprehensive_scan(
            symbols=symbols if symbols else None,
            sector_filter=sector_filter,
            min_score=min_score)

        # Track performance for enhanced scan results
        tracked_count = 0
        if results and results.get('opportunities'):
            from performance_tracker import PerformanceTracker
            tracker = PerformanceTracker()

            for opportunity in results.get('opportunities', []):
                try:
                    if 'symbol' in opportunity and 'best_option' in opportunity:
                        track_id = tracker.track_option_performance(
                            opportunity['symbol'], opportunity['best_option'],
                            opportunity.get('trading_plan', {}), scan_date)
                        tracked_count += 1
                        print(
                            f"📊 Started tracking {opportunity['symbol']}: {trackid}"
                        )
                except Exception as e:
                    print(
                        f"⚠️ Failed to track {opportunity.get('symbol', 'unknown')}: {e}"
                    )

        return jsonify({
            "status":
            "success",
            "scan_date":
            scan_date,
            "opportunities_found":
            len(results.get('opportunities', [])),
            "top_picks":
            results.get('top_picks', [])[:10],
            "market_regime":
            results.get('market_regime', {}),
            "performance_tracking":
            f"Now tracking {tracked_count} options"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Enhanced scan failed: {str(e)}"
        }), 500


@app.route("/explosive-scan", methods=["GET", "POST"])
def run_explosive_scan():
    """Run the explosive options scanner optimized for your API plan"""
    try:
        from explosive_options_scanner import ExplosiveOptionsScanner
        from scanner_core import auto_discover_optionable_universe

        # Get parameters
        if request.method == 'POST' and request.is_json:
            data = request.get_json()
            scan_type = data.get('scan_type', 'comprehensive')
            symbols = data.get('symbols', None)
            filters = data.get('filters', {})
            market_data = data.get('market_data', {})
            max_symbols = data.get('max_symbols',
                                   400)  # Default limit for efficiency
        else:
            scan_type = request.args.get('scan_type', 'comprehensive')
            symbols = request.args.getlist('symbols') or None
            max_symbols = int(request.args.get('max_symbols', 400))
            filters = {
                'min_price': float(request.args.get('min_price', 0.05)),
                'max_price': float(request.args.get('max_price', 5.00)),
                'min_delta': float(request.args.get('min_delta', 0.15)),
                'max_delta': float(request.args.get('max_delta', 0.35)),
                'min_days': int(request.args.get('min_days', 1)),
                'max_days': int(request.args.get('max_days', 21))
            }
            market_data = {}

        # Auto-discover symbols for explosive scan if none provided
        if not symbols:
            limit = int(os.getenv("EXPLOSIVE_DISCOVERY_LIMIT", 300))
            include_etfs = (os.getenv("INCLUDE_ETFS", "1") == "1")
            symbols = auto_discover_optionable_universe(limit=limit, include_etfs=include_etfs)
            symbols = sorted(set([s.upper() for s in symbols if s]))
            print(f"🔍 Auto-discovered {len(symbols)} symbols for explosive scan (limit={limit}, include_etfs={include_etfs})")
        # Initialize scanner
        scanner = ExplosiveOptionsScanner(os.getenv('ALPHA_VANTAGE_API_KEY'))

        # Limit symbols if provided to stay within API constraints
        if symbols and len(symbols) > max_symbols:
            symbols = symbols[:max_symbols]
            print(f"⚡ Limited to {max_symbols} symbols for API efficiency")

        # Run scan
        results = scanner.run_explosive_scan(
            symbols=symbols,
            scan_type=scan_type,
            filters=filters,
            market_data=market_data
            if request.method == 'POST' and request.is_json else None)

        # Track performance for all opportunities found
        tracked_count = 0
        if results and results.get('opportunities'):
            from performance_tracker import PerformanceTracker
            tracker = PerformanceTracker()

            for symbol, opportunity_data in results['opportunities'].items():
                try:
                    best_option = opportunity_data.get('best_opportunity', {})
                    trading_plan = opportunity_data.get('trading_plan', {})

                    if best_option and trading_plan:
                        track_id = tracker.track_option_performance(
                            symbol, best_option, trading_plan,
                            datetime.now().strftime('%Y-%m-%d'))
                        tracked_count += 1
                        print(f"📊 Started tracking {symbol}: {track_id}")
                except Exception as e:
                    print(f"⚠️ Failed to track {symbol}: {e}")

        return jsonify({
            "status":
            "success",
            "scan_type":
            scan_type,
            "api_optimization":
            "Historical options + Yahoo Finance fallback",
            "symbols_scanned":
            results['scan_metadata']['symbols_scanned'],
            "opportunities_found":
            len(results['opportunities']),
            "top_picks":
            results['top_picks'][:10],
            "earnings_opportunities":
            len(results['by_category']['earnings_plays']),
            "api_calls_saved":
            "Using bulk quotes + historical options",
            "performance_tracking":
            f"Now tracking {tracked_count} options",
            "summary":
            results['summary']
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Explosive scan failed: {str(e)}"
        }), 500


@app.route("/monitor-positions", methods=["GET"])
def monitor_positions():
    """Monitor active positions with adaptive system"""
    try:
        from explosive_options_scanner import ExplosiveOptionsScanner

        scanner = ExplosiveOptionsScanner(os.getenv('ALPHA_VANTAGE_API_KEY'))
        monitoring_results = scanner.monitor_active_positions()

        return jsonify({
            "status":
            "success",
            "positions_monitored":
            monitoring_results['positions_monitored'],
            "alerts":
            monitoring_results['alerts'],
            "exit_signals":
            monitoring_results['exit_signals'],
            "summary_actions":
            monitoring_results['summary_actions']
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Monitoring encountered an error: {str(e)}"
        }), 500


@app.route("/market-regime", methods=["GET"])
def get_market_regime():
    """Retrieve the current market regime and associated trading adjustments"""
    try:
        from adaptive_market_monitor import AdaptiveMarketMonitor

        monitor = AdaptiveMarketMonitor(os.getenv('ALPHA_VANTAGE_API_KEY'))
        regime_update = monitor.update_market_regime()

        return jsonify({
            "status":
            "success",
            "current_regime":
            regime_update['current_regime'],
            "changes":
            regime_update['changes'],
            "trading_adjustments":
            regime_update['trading_adjustments'],
            "timestamp":
            regime_update['timestamp']
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Market regime analysis failed: {str(e)}"
        }), 500


@app.route("/performance/live", methods=["GET"])
def get_live_performance():
    """Get real-time performance of all tracked options"""
    try:
        from performance_tracker import PerformanceTracker

        tracker = PerformanceTracker()
        performance_data = tracker.load_performance_data()

        # Get current performance for active positions
        active_positions = []
        expired_positions = []
        profitable_positions = []
        losing_positions = []

        for track_id, data in performance_data.items():
            position_info = {
                'track_id':
                track_id,
                'symbol':
                data['symbol'],
                'prediction_date':
                data['prediction_date'],
                'option_type':
                data['option_details'].get('type', 'N/A'),
                'strike':
                data['option_details'].get('strike', 'N/A'),
                'expiration':
                data['option_details'].get('expiration', 'N/A'),
                'entry_price':
                data['option_details'].get('entry_price', 0),
                'confluence_score':
                data['option_details'].get('confluence_score', 0),
                'max_profit':
                data.get('max_profit', 0),
                'max_loss':
                data.get('max_loss', 0),
                'final_outcome':
                data.get('final_outcome'),
                'days_tracked':
                data.get('days_tracked', 0)
            }

            if data.get('final_outcome') is None:
                active_positions.append(position_info)
            elif data.get('final_outcome') == 'EXPIRED':
                expired_positions.append(position_info)
            elif data.get('max_profit', 0) > 0:
                profitable_positions.append(position_info)
            else:
                losing_positions.append(position_info)

        # Sort by max profit/loss
        profitable_positions.sort(key=lambda x: x['max_profit'], reverse=True)
        losing_positions.sort(key=lambda x: x['max_loss'])

        return jsonify({
            "status": "success",
            "summary": {
                "total_tracked": len(performance_data),
                "active_positions": len(active_positions),
                "expired_positions": len(expired_positions),
                "profitable_count": len(profitable_positions),
                "losing_count": len(losing_positions)
            },
            "active_positions": active_positions[:20],  # Top 20
            "top_performers": profitable_positions[:10],
            "worst_performers": losing_positions[:10],
            "recently_expired": expired_positions[-10:]  # Last 10 expired
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error getting live performance: {str(e)}"
        }), 500


@app.route("/performance/update-now", methods=["POST", "GET"])
def update_performance_now():
    """Manually trigger performance update for all tracked options"""
    try:
        from performance_tracker import PerformanceTracker

        tracker = PerformanceTracker()
        updated_count = tracker.update_daily_performance()

        # Get quick stats after update
        metrics = tracker.calculate_performance_metrics()

        return jsonify({
            "status": "success",
            "message": f"Performance updated for {updated_count} options",
            "updated_count": updated_count,
            "quick_stats": {
                "total_predictions": metrics.get('total_predictions', 0),
                "win_rate": f"{metrics.get('win_rate', 0):.1f}%",
                "targets_hit": metrics.get('targets_hit', 0),
                "stops_hit": metrics.get('stops_hit', 0),
                "still_active": metrics.get('still_active', 0)
            },
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error updating performance: {str(e)}"
        }), 500


@app.route("/performance/position/<track_id>", methods=["GET"])
def get_position_details(track_id):
    """Get detailed tracking information for a specific position"""
    try:
        from performance_tracker import PerformanceTracker

        tracker = PerformanceTracker()
        performance_data = tracker.load_performance_data()

        if track_id not in performance_data:
            return jsonify({
                "status": "error",
                "message": f"Position {track_id} not found"
            }), 404

        position_data = performance_data[track_id]

        # Calculate additional metrics
        daily_tracking = position_data.get('daily_tracking', {})
        if daily_tracking:
            dates = sorted(daily_tracking.keys())
            price_history = [daily_tracking[date]['price'] for date in dates]
            pnl_history = [
                daily_tracking[date]['pnl_percent'] for date in dates
            ]
        else:
            dates = []
            price_history = []
            pnl_history = []

        return jsonify({
            "status": "success",
            "position_details": position_data,
            "price_history": {
                "dates": dates,
                "prices": price_history,
                "pnl_percentages": pnl_history
            },
            "current_status": {
                "is_active": position_data.get('final_outcome') is None,
                "days_held": len(daily_tracking),
                "best_day": max(pnl_history) if pnl_history else 0,
                "worst_day": min(pnl_history) if pnl_history else 0
            }
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error getting position details: {str(e)}"
        }), 500


@app.route("/backtest/run", methods=["GET", "POST"])
def run_comprehensive_backtest():
    """Run comprehensive backtest on all historical trades"""
    try:
        from backtest_engine import AdvancedBacktester

        # Get parameters
        if request.method == 'POST' and request.is_json:
            data = request.get_json()
            include_human_readable = data.get('include_human_readable', False)
            min_score_filter = data.get('min_score_filter', 0)
            days_lookback = data.get('days_lookback', 365)
        else:
            include_human_readable = request.args.get(
                'include_human_readable', 'false').lower() == 'true'
            min_score_filter = float(request.args.get('min_score_filter', 0))
            days_lookback = int(request.args.get('days_lookback', 365))

        print("🚀 Starting comprehensive backtest...")

        # Initialize backtester
        backtester = AdvancedBacktester()

        # Load historical trades from JSON files
        trades = backtester.load_all_historical_trades()

        # Optionally include human readable file
        if include_human_readable:
            human_file = "./TradingPlans/human_readable_plans.txt"
            if os.path.exists(human_file):
                human_trades = backtester.parse_human_readable_file(human_file)
                trades.extend(human_trades)
                print(
                    f"Added {len(human_trades)} trades from human readable file"
                )

        if not trades:
            return jsonify({
                "status": "error",
                "message": "No historical trades found to backtest"
            }), 404

        # Filter by date and score if specified
        filtered_trades = []
        cutoff_date = datetime.now() - timedelta(days=days_lookback)

        for trade in trades:
            # Check date filter
            trade_date = pd.to_datetime(trade.prediction_date)
            if trade_date < cutoff_date:
                continue

            # Check score filter
            score = getattr(trade, 'confluence_score', 0) or getattr(
                trade, 'score', 0)
            if score < min_score_filter:
                continue

            filtered_trades.append(trade)

        backtester.trades = filtered_trades

        if not filtered_trades:
            return jsonify({
                "status":
                "error",
                "message":
                f"No trades match filters (min_score: {min_score_filter}, days_lookback: {days_lookback})"
            }), 404

        # Run backtest
        backtester.backtest_all_trades()

        # Analyze results
        results_df = backtester.analyze_results()

        # Generate summary for API response
        valid_results = [
            r for r in backtester.results if r.get('status') not in
            ['ERROR', 'NO_STOCK_DATA', 'FUTURE_TRADE']
        ]

        if valid_results:
            df = pd.DataFrame([{
                'hit_initial':
                r.get('hit_initial', False),
                'hit_final':
                r.get('hit_final', False),
                'hit_stop':
                r.get('hit_stop', False),
                'profit_loss':
                r.get('profit_loss', 0),
                'profit_loss_pct':
                r.get('profit_loss_pct', 0),
                'confluence_score':
                getattr(r['trade'], 'confluence_score', 0)
                or getattr(r['trade'], 'score', 0)
            } for r in valid_results])

            # Calculate summary statistics
            win_rate = (df['hit_initial'] | df['hit_final']).mean() * 100
            initial_target_rate = df['hit_initial'].mean() * 100
            final_target_rate = df['hit_final'].mean() * 100
            stop_loss_rate = df['hit_stop'].mean() * 100
            avg_return = df['profit_loss_pct'].mean()
            total_pnl = df['profit_loss'].sum()

            # Best performing score threshold
            high_score_trades = df[df['confluence_score'] >= 8]
            high_score_win_rate = 0
            if len(high_score_trades) > 0:
                high_score_win_rate = (
                    high_score_trades['hit_initial']
                    | high_score_trades['hit_final']).mean() * 100

        else:
            win_rate = initial_target_rate = final_target_rate = stop_loss_rate = 0
            avg_return = total_pnl = high_score_win_rate = 0

        return jsonify({
            "status":
            "success",
            "backtest_summary": {
                "total_trades_analyzed":
                len(backtester.trades),
                "valid_results":
                len(valid_results),
                "filters_applied": {
                    "min_score_filter": min_score_filter,
                    "days_lookback": days_lookback,
                    "include_human_readable": include_human_readable
                },
                "performance_metrics": {
                    "win_rate": round(win_rate, 2),
                    "initial_target_rate": round(initial_target_rate, 2),
                    "final_target_rate": round(final_target_rate, 2),
                    "stop_loss_rate": round(stop_loss_rate, 2),
                    "average_return_pct": round(avg_return, 2),
                    "total_pnl": round(total_pnl, 2),
                    "high_score_win_rate": round(high_score_win_rate, 2)
                },
                "optimization_insights": [
                    f"Trades with scores >= 8 show {high_score_win_rate:.1f}% win rate",
                    f"Overall win rate: {win_rate:.1f}% across {len(valid_results)} trades",
                    f"Consider focusing on confluence scores >= 8 for better performance",
                    f"Stop loss hit rate: {stop_loss_rate:.1f}% - consider tighter risk management"
                ]
            },
            "timestamp":
            datetime.now().isoformat(),
            "files_generated": [
                "backtest_results_[timestamp].csv",
                "backtest_summary_[timestamp].json"
            ]
        })

    except Exception as e:
        import traceback
        return jsonify({
            "status": "error",
            "message": f"Backtesting failed: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500


@app.route("/backtest/optimize-thresholds", methods=["GET", "POST"])
def optimize_scanner_thresholds():
    """Optimize scanner thresholds based on backtest results"""
    try:
        from backtest_engine import AdvancedBacktester

        print(
            "🎯 Optimizing scanner thresholds based on historical performance..."
        )

        # Run backtest first
        backtester = AdvancedBacktester()
        trades = backtester.load_all_historical_trades()

        if not trades:
            return jsonify({
                "status": "error",
                "message": "No historical data for optimization"
            }), 404

        backtester.backtest_all_trades()

        # Analyze for optimal thresholds
        valid_results = [
            r for r in backtester.results if r.get('status') not in
            ['ERROR', 'NO_STOCK_DATA', 'FUTURE_TRADE']
        ]

        if not valid_results:
            return jsonify({
                "status": "error",
                "message": "No valid results for optimization"
            }), 404

        # Find optimal thresholds
        df = pd.DataFrame([{
            'confluence_score':
            getattr(r['trade'], 'confluence_score', 0)
            or getattr(r['trade'], 'score', 0),
            'delta':
            getattr(r['trade'], 'delta', 0) or 0,
            'gamma':
            getattr(r['trade'], 'gamma', 0) or 0,
            'theta':
            getattr(r['trade'], 'theta', 0) or 0,
            'successful':
            r.get('hit_initial', False) or r.get('hit_final', False)
        } for r in valid_results])

        # Find optimal score threshold (maximize win rate with reasonable sample size)
        optimal_thresholds = {}

        for threshold in [5, 6, 7, 8, 9]:
            subset = df[df['confluence_score'] >= threshold]
            if len(subset) >= 10:  # Minimum sample size
                win_rate = subset['successful'].mean()
                optimal_thresholds[threshold] = {
                    'win_rate': win_rate * 100,
                    'sample_size': len(subset)
                }

        # Find best threshold
        best_threshold = max(optimal_thresholds.keys(),
                             key=lambda x: optimal_thresholds[x]['win_rate'])

        # Find optimal Greeks ranges
        successful_trades = df[df['successful'] == True]
        delta_mean = successful_trades['delta'].mean()
        delta_std = successful_trades['delta'].std()

        gamma_mean = successful_trades['gamma'].mean()
        gamma_std = successful_trades['gamma'].std()

        recommendations = {
            "optimal_confluence_threshold":
            best_threshold,
            "confluence_threshold_analysis":
            optimal_thresholds,
            "optimal_delta_range":
            [max(0, delta_mean - delta_std),
             min(1, delta_mean + delta_std)],
            "optimal_gamma_range":
            [max(0, gamma_mean - gamma_std), gamma_mean + gamma_std],
            "implementation_suggestions": [
                f"Set minimum confluence score to {best_threshold} (current best performing)",
                f"Focus on delta range {delta_mean-delta_std:.3f} to {delta_mean+delta_std:.3f}",
                f"Prioritize gamma > {gamma_mean:.3f} for explosive potential",
                "Consider implementing dynamic thresholds based on market volatility"
            ]
        }

        return jsonify({
            "status": "success",
            "optimization_results": recommendations,
            "sample_size": len(df),
            "successful_trades": len(successful_trades),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Optimization failed: {str(e)}"
        }), 500


@app.route("/test-api-key", methods=["GET"])
def test_api_key():
    """Test Alpha Vantage API key functionality"""
    try:
        import requests

        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            return jsonify({
                "status":
                "error",
                "message":
                "ALPHA_VANTAGE_API_KEY environment variable not found"
            }), 500

        # Test with a simple quote request
        url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={api_key}'
        response = requests.get(url, timeout=10)
        data = response.json()

        if 'Global Quote' in data:
            return jsonify({
                "status": "success",
                "message": "Alpha Vantage API key is working correctly",
                "sample_data": {
                    "symbol": data['Global Quote']['01. symbol'],
                    "price": data['Global Quote']['05. price'],
                    "change": data['Global Quote']['09. change']
                },
                "api_key_masked": f"{api_key[:8]}...{api_key[-4:]}"
            })
        elif 'Information' in data:
            return jsonify({
                "status": "warning",
                "message": f"API limit issue: {data['Information']}",
                "api_key_masked": f"{api_key[:8]}...{api_key[-4:]}"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Unexpected API response",
                "response": data
            }), 500

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"API test failed: {str(e)}"
        }), 500


@app.route("/test-bulk-quotes", methods=["GET"])
def test_bulk_quotes():
    """Test Alpha Vantage bulk quotes functionality"""
    try:
        import requests
        from immediate_fixes import process_alpha_vantage_bulk_response

        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            return jsonify({
                "status":
                "error",
                "message":
                "ALPHA_VANTAGE_API_KEY environment variable not found"
            }), 500

        # Test with a small set of symbols
        test_symbols = "AAPL,MSFT,GOOGL,TSLA,NVDA"
        url = f'https://www.alphavantage.co/query?function=REALTIME_BULK_QUOTES&symbol={test_symbols}&apikey={api_key}'

        print(f"🧪 Testing bulk quotes with URL: {url}")
        response = requests.get(url, timeout=30)
        data = response.json()

        print(f"🧪 Response status: {response.status_code}")
        print(
            f"🧪 Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}"
        )
        print(f"🧪 Raw response sample: {str(data)[:500]}...")

        # Try to parse the response
        parsed = process_alpha_vantage_bulk_response(data)

        return jsonify({
            "status":
            "success" if len(parsed) > 0 else "no_data",
            "test_url":
            url,
            "response_keys":
            list(data.keys()) if isinstance(data, dict) else [],
            "parsed_symbols":
            len(parsed),
            "parsed_data":
            parsed,
            "raw_response_sample":
            str(data)[:1000],
            "api_key_masked":
            f"{api_key[:8]}...{api_key[-4:]}"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Bulk quotes test failed: {str(e)}"
        }), 500


@app.route("/test-earnings-verbose", methods=["GET"])
def test_earnings_verbose():
    """Test endpoint to debug earnings discovery with full verbose output"""
    try:
        from enhanced_scanner import discover_pre_earnings_stocks

        print("🧪 Running verbose earnings test...")
        pre_earnings_stocks = discover_pre_earnings_stocks(verbose=True)

        return jsonify({
            "status":
            "success",
            "test_type":
            "verbose_earnings_discovery",
            "candidates_found":
            len(pre_earnings_stocks),
            "candidates":
            pre_earnings_stocks,
            "message":
            f"Verbose test complete - found {len(pre_earnings_stocks)} candidates"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Verbose test failed: {str(e)}"
        }), 500


@app.route("/earnings-calendar", methods=["GET"])
def get_earnings_calendar():
    """Get raw earnings calendar data from Alpha Vantage"""
    try:
        import os
        import requests
        from datetime import datetime

        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            return jsonify({
                "status": "error",
                "message": "ALPHA_VANTAGE_API_KEY not configured"
            }), 500

        # Get horizon parameter (default 3month)
        horizon = request.args.get('horizon', '3month')
        show_all = request.args.get('show_all', 'false').lower() == 'true'

        url = f'https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&horizon={horizon}&apikey={api_key}'

        print(
            f"📅 Fetching earnings calendar from Alpha Vantage (horizon: {horizon})..."
        )
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Parse CSV response
        lines = response.text.strip().split('\n')

        if len(lines) < 2:
            return jsonify({
                "status": "no_data",
                "message": "No earnings data returned from Alpha Vantage",
                "raw_response": response.text[:500]
            })

        # Parse CSV into JSON
        headers = lines[0].split(',')
        earnings_data = []

        for line in lines[1:]:
            fields = line.split(',')
            if len(fields) >= len(headers):
                earnings_record = {}
                for i, header in enumerate(headers):
                    earnings_record[header.strip().strip(
                        '"')] = fields[i].strip().strip('"')
                earnings_data.append(earnings_record)

        # Filter for next 21 days
        current_date = datetime.now().date()
        upcoming_earnings = []
        alphabet_breakdown = {}

        for record in earnings_data:
            try:
                symbol = record.get('symbol', '')
                earnings_date_str = record.get('reportDate', '')
                if earnings_date_str and symbol:
                    earnings_date = datetime.strptime(earnings_date_str,
                                                      '%Y-%m-%d').date()
                    days_to_earnings = (earnings_date - current_date).days

                    if 0 <= days_to_earnings <= 21:
                        record['days_to_earnings'] = days_to_earnings
                        upcoming_earnings.append(record)

                        # Track alphabet distribution
                        first_letter = symbol[0] if symbol else 'Unknown'
                        alphabet_breakdown[
                            first_letter] = alphabet_breakdown.get(
                                first_letter, 0) + 1
            except:
                continue

        # Sort upcoming earnings by date
        upcoming_earnings.sort(key=lambda x: x.get('days_to_earnings', 999))

        response_data = {
            "status": "success",
            "horizon": horizon,
            "total_records": len(earnings_data),
            "upcoming_21_days": len(upcoming_earnings),
            "alphabet_breakdown": alphabet_breakdown,
            "query_time": datetime.now().isoformat()
        }

        if show_all:
            response_data["all_upcoming"] = upcoming_earnings
        else:
            response_data["sample_upcoming"] = upcoming_earnings[:20]
            response_data["major_stocks_upcoming"] = [
                record for record in upcoming_earnings
                if record.get('symbol', '') in [
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
                    'NFLX', 'JPM', 'BAC'
                ]
            ]

        return jsonify(response_data)

    except Exception as e:
        return jsonify({
            "status":
            "error",
            "message":
            f"Failed to fetch earnings calendar: {str(e)}"
        }), 500


@app.route('/api/jpm-explosion-hunter', methods=['GET', 'POST'])
def jpm_explosion_hunter():
    """
    Phase 2: Find options matching the JPM explosion pattern from your example contract.
    This takes results from explosive-earnings-combo and filters for JPM-like setups.

    JPM Example Pattern:
    - Strike 315 CALL expiring 2025-07-25 (17 days out)
    - Delta: 0.15696, Gamma: 0.01411, Theta: -0.09836
    - Volume: 121, OI: 129, IV: 0.23438
    - Price: $1.52, Current stock: $293.725
    """
    try:
        # Get parameters
        if request.method == 'POST' and request.is_json:
            data = request.get_json()
            source_symbols = data.get('symbols', [])
            similarity_threshold = float(data.get('similarity_threshold', 0.7))
        else:
            source_symbols = request.args.getlist('symbols')
            similarity_threshold = float(
                request.args.get('similarity_threshold', 0.7))

        print("🎯 JPM EXPLOSION HUNTER - PHASE 2 PATTERN MATCHING")
        print("=" * 60)
        print(f"📋 Target Pattern: JPM 315C exp 7/25 @ $1.52")
        print(f"🔍 Delta: 0.157, Gamma: 0.014, Theta: -0.098")
        print(f"📊 Volume/OI: 121/129, IV: 0.234")
        print(f"🎯 Similarity Threshold: {similarity_threshold:.1%}")

        # NET reference pattern
        net_reference = {
            'delta': 0.15696,
            'gamma': 0.01411,
            'theta': -0.09836,
            'price': 1.52,
            'iv': 0.23438,
            'days_to_expiry': 17
        }

        # If no symbols provided, get from most recent explosive scan results
        if not source_symbols:
            print("📂 Loading from recent explosive scan results...")
            import glob
            pattern = os.path.join('./TradingPlans', 'explosive_scan_*.json')
            files = glob.glob(pattern)
            if files:
                latest_file = max(files, key=os.path.getctime)
                with open(latest_file, 'r') as f:
                    scan_data = json.load(f)
                    source_symbols = list(
                        scan_data.get('opportunities', {}).keys())
                    print(
                        f"📊 Loaded {len(source_symbols)} symbols from {os.path.basename(latest_file)}"
                    )
            else:
                # Fallback to common symbols
                source_symbols = [
                    'SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
                    'NVDA', 'META', 'JPM', 'BAC', 'WFC'
                ]
                print(f"⚠️ No scan results found - using fallback symbols")

        print(f"🔍 Analyzing options from {len(source_symbols)} symbols...")

        jpm_matches = []
        symbols_analyzed = 0

        for symbol in source_symbols[:30]:  # Limit for performance
            try:
                print(f"📊 Analyzing {symbol}...")

                # Get options data for this symbol
                from explosive_options_scanner import ExplosiveOptionsScanner
                scanner = ExplosiveOptionsScanner(
                    os.getenv('ALPHA_VANTAGE_API_KEY'))
                options_data = scanner._fetch_all_options(symbol)

                if options_data is None or options_data.empty:
                    print(f"⚠️ No options data for {symbol}")
                    continue

                symbols_analyzed += 1

                # Analyze each option for JPM similarity
                for idx, option in options_data.iterrows():
                    try:
                        # Extract option data
                        option_dict = {
                            'delta':
                            abs(float(option.get('delta', 0))),
                            'gamma':
                            float(option.get('gamma', 0)),
                            'theta':
                            float(option.get('theta', 0)),
                            'price':
                            float(
                                option.get('mark', option.get('lastPrice',
                                                              0))),
                            'iv':
                            float(option.get('impliedVolatility', 0)),
                            'volume':
                            float(option.get('volume', 0)),
                            'open_interest':
                            float(
                                option.get('open_interest',
                                           option.get('openInterest', 0))),
                            'strike':
                            float(option.get('strike', 0)),
                            'type':
                            str(option.get('type', 'call')),
                            'expiration':
                            str(option.get('expiration', ''))
                        }

                        # Calculate days to expiry
                        try:
                            exp_date = pd.to_datetime(
                                option_dict['expiration'])
                            option_dict['days_to_expiry'] = max(
                                1, (exp_date - pd.Timestamp.now()).days)
                        except:
                            option_dict['days_to_expiry'] = 30  # Default

                        # Calculate similarity score based on Greeks and other factors
                        def calculate_similarity(option, reference):
                            try:
                                # Extract values safely
                                delta_diff = abs(
                                    float(option.get('delta', 0)) -
                                    reference['delta'])
                                gamma_diff = abs(
                                    float(option.get('gamma', 0)) -
                                    reference['gamma'])

                                # Theta comparison: match both sign and magnitude
                                option_theta = float(option.get('theta', 0))
                                ref_theta = reference['theta']

                                # If signs are different, heavily penalize
                                if (option_theta > 0) != (ref_theta > 0):
                                    theta_diff = abs(option_theta) + abs(
                                        ref_theta
                                    )  # Heavy penalty for sign mismatch
                                else:
                                    theta_diff = abs(
                                        option_theta - ref_theta
                                    )  # Normal difference for same sign

                                price_diff = abs(
                                    float(
                                        option.get('mark',
                                                   option.get('lastPrice', 0)))
                                    - reference['price'])
                                iv_diff = abs(
                                    float(option.get('impliedVolatility', 0)) -
                                    reference['iv'])
                                days_diff = abs(
                                    int(option.get('days_to_expiry', 0)) -
                                    reference['days_to_expiry'])

                                # Weighted similarity calculation (lower is more similar)
                                similarity = (
                                    delta_diff * 25 +  # Delta weight
                                    gamma_diff * 50 +  # Gamma weight  
                                    theta_diff * 15
                                    +  # Theta weight (now preserves sign)
                                    (price_diff /
                                     max(reference['price'], 0.01)) * 10
                                    +  # Price % diff
                                    iv_diff * 20 +  # IV weight
                                    (days_diff / 30) *
                                    5  # Days weight (normalized)
                                )

                                # Convert to percentage (higher = more similar)
                                similarity_score = max(0,
                                                       (1 - similarity) * 100)
                                return min(100, similarity_score)

                            except Exception as e:
                                print(f"Error calculating similarity: {e}")
                                return 0

                        # Calculate similarity score
                        similarity_score = calculate_net_similarity(
                            option_dict, net_reference)

                        # Check if it meets threshold
                        if similarity_score >= similarity_threshold:
                            match = {
                                'symbol': symbol,
                                'option': option_dict,
                                'similarity_score': round(similarity_score, 3),
                                'contract_symbol':
                                f"{symbol} {option_dict['strike']} {option_dict['type'].upper()} exp {option_dict['expiration'][:10]}",
                                'comparison': {
                                    'delta_diff':
                                    abs(option_dict['delta'] -
                                        net_reference['delta']),
                                    'gamma_diff':
                                    abs(option_dict['gamma'] -
                                        net_reference['gamma']),
                                    'theta_diff':
                                    abs(option_dict['theta'] -
                                        net_reference['theta']),
                                    'price_diff':
                                    abs(option_dict['price'] -
                                        net_reference['price']),
                                    'iv_diff':
                                    abs(option_dict['iv'] -
                                        net_reference['iv']),
                                    'days_diff':
                                    abs(option_dict['days_to_expiry'] -
                                        net_reference['days_to_expiry'])
                                }
                            }
                            jpm_matches.append(match)
                            print(
                                f"✅ JPM-like match found: {match['contract_symbol']} (similarity: {similarity_score:.1%})"
                            )

                    except Exception as e:
                        continue  # Skip problematic options

            except Exception as e:
                print(f"⚠️ Error analyzing {symbol}: {e}")
                continue

        # Sort matches by similarity score
        jpm_matches.sort(key=lambda x: x['similarity_score'], reverse=True)

        print(f"\n🎯 JPM EXPLOSION HUNTER COMPLETE")
        print(f"📊 Symbols Analyzed: {symbols_analyzed}")
        print(f"🔥 JPM-like Matches Found: {len(jpm_matches)}")

        if jpm_matches:
            print(
                f"🏆 Best Match: {jpm_matches[0]['contract_symbol']} ({jpm_matches[0]['similarity_score']:.1%} similar)"
            )

        # Save JPM hunter results to dedicated file
        timestamp = datetime.now().strftime('%H%M%S')
        date_str = datetime.now().strftime('%Y-%m-%d')
        jpm_filename = os.path.join(
            './TradingPlans',
            f'jpm_explosion_hunter_{date_str}_{timestamp}.json')

        jpm_results = {
            "scan_metadata": {
                "timestamp": datetime.now().isoformat(),
                "scan_type": "jpm_explosion_hunter",
                "net_reference_pattern": net_reference,
                "similarity_threshold": similarity_threshold,
                "source_file":
                latest_file if not source_symbols else "custom_symbols",
                "total_symbols": len(source_symbols),
                "symbols_analyzed": symbols_analyzed
            },
            "jpm_matches":
            jpm_matches,
            "top_10_matches":
            jpm_matches[:10],
            "summary":
            f"Found {len(jpm_matches)} NET-like options from {symbols_analyzed} symbols analyzed"
        }

        # Save results
        try:
            with open(jpm_filename, 'w') as f:
                json.dump(jpm_results, f, indent=2, default=str)
            print(f"💾 JPM hunter results saved to {jpm_filename}")
        except Exception as e:
            print(f"⚠️ Could not save JPM results: {e}")

        return jsonify({
            "status":
            "success",
            "total_symbols":
            len(source_symbols),
            "symbols_analyzed":
            symbols_analyzed,
            "net_matches":
            jpm_matches[:10],  # Top 10 matches
            "net_reference_pattern":
            net_reference,
            "similarity_threshold":
            similarity_threshold,
            "results_saved_to":
            jpm_filename,
            "summary":
            f"Found {len(jpm_matches)} NET-like options from {symbols_analyzed} symbols analyzed"
        })

    except Exception as e:
        import traceback
        return jsonify({
            "status": "error",
            "message": f"JPM explosion hunter failed: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500


def calculate_net_similarity(option_dict, net_reference):
    """Calculate similarity score between option and NET reference pattern"""
    try:
        # Optimized similarity weights (total = 1.0) - Based on NET explosion characteristics
        weights = {
            'delta': 0.30,  # Most important - directional exposure
            'price': 0.25,  # Critical - entry cost and risk
            'gamma': 0.20,  # Important - acceleration potential
            'days_to_expiry': 0.15,  # Key - time decay window
            'theta': 0.07,  # Moderate - time decay rate
            'iv': 0.03  # Least - already captured in price
        }

        total_similarity = 0.0

        # Calculate similarity for each metric (1.0 = perfect match, 0.0 = very different)
        for metric, weight in weights.items():
            if metric in option_dict and metric in net_reference:
                option_val = option_dict[metric]
                ref_val = net_reference[metric]

                # Calculate percentage difference
                if ref_val != 0:
                    diff = abs(option_val - ref_val) / abs(ref_val)
                else:
                    diff = abs(option_val) if option_val != 0 else 0

                # Convert to similarity (closer to 0 diff = higher similarity)
                similarity = max(0, 1 - diff)

                # Apply tolerance - some metrics can be more different
                if metric == 'price':
                    similarity = max(0, 1 -
                                     (diff / 2))  # Allow more price variation
                elif metric == 'days_to_expiry':
                    similarity = max(0, 1 -
                                     (diff / 3))  # Allow more time variation

                total_similarity += similarity * weight

        return min(1.0, total_similarity)  # Cap at 1.0

    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0


@app.route("/explosive-earnings-combo", methods=["GET", "POST"])
def explosive_earnings_combo():
    """
            True two-phase scan:
            1. Finds candidates with the ExplosiveOptionsScanner.
            2. Performs deep multi-timeframe analysis with scanner_core.
            3. Combines both scores to find the best opportunities.
            """
    try:  # The stray backtick has been removed from this line
        # Get parameters from request
        if request.method == 'POST' and request.is_json:
            data = request.get_json()
            min_explosive_score = float(data.get('min_explosive_score', 40))
            min_confluence_score = float(data.get('min_confluence_score', 6.0))
            filters = data.get('filters', {})
        else:  # GET request
            min_explosive_score = float(
                request.args.get('min_explosive_score', 40))
            min_confluence_score = float(
                request.args.get('min_confluence_score', 6.0))
            filters = {
                'min_price': float(request.args.get('min_price', 0.05)),
                'max_price': float(request.args.get('max_price', 5.00)),
                'min_delta': float(request.args.get('min_delta', 0.10)),
                'max_delta': float(request.args.get('max_delta', 0.40)),
                'min_days': int(request.args.get('min_days', 1)),
                'max_days': int(request.args.get('max_days', 30))
            }

        print("🚀 TWO-PHASE EXPLOSIVE EARNINGS SCAN")
        print("=" * 60)

        from explosive_options_scanner import ExplosiveOptionsScanner
        from scanner_core import run_scanner

        # --- PHASE 1: Find candidates with unusual options activity ---
        print(
            "\n📊 PHASE 1: Discovering candidates with ExplosiveOptionsScanner..."
        )
        explosive_scanner = ExplosiveOptionsScanner(
            os.getenv('ALPHA_VANTAGE_API_KEY'))
        explosive_results = explosive_scanner.run_explosive_scan(
            scan_type='earnings', filters=filters)

        candidates = []
        for symbol, data in explosive_results.get('opportunities', {}).items():
            score = data.get('best_opportunity', {}).get('total_score', 0)
            is_earnings = data.get('market_data',
                                   {}).get('earnings_info',
                                           {}).get('is_pre_earnings', False)
            if score >= min_explosive_score and is_earnings:
                candidates.append(symbol)

        print(
            f"✅ PHASE 1 COMPLETE: Found {len(candidates)} candidates meeting explosive criteria."
        )
        if not candidates:
            return jsonify({
                "status": "no-results",
                "message": "Phase 1 found no suitable candidates."
            })

        # --- PHASE 2: Run deep technical analysis on candidates ---
        print(
            f"\n🔬 PHASE 2: Running scanner_core analysis on {len(candidates)} candidates..."
        )
        scanner_core_results = run_scanner(
            symbols=candidates,
            min_delta=filters.get('min_delta', 0.10),
            max_delta=filters.get('max_delta', 0.40),
            min_price=filters.get('min_price', 0.05),
            max_price=filters.get('max_price', 5.00),
            time_to_expiry_range=(filters.get('min_days',
                                              1), filters.get('max_days', 30)))
        print(
            f"✅ PHASE 2 COMPLETE: Analyzed {len(scanner_core_results)} symbols."
        )

        # --- FINAL STEP: Combine scores and find true explosive setups ---
        final_opportunities = []
        for symbol, core_data in scanner_core_results.items():
            confluence_score = core_data.get('confluence', {}).get('score', 0)

            if confluence_score >= min_confluence_score:
                explosive_data = explosive_results['opportunities'].get(
                    symbol, {})
                explosive_score = explosive_data.get('best_opportunity',
                                                     {}).get('total_score', 0)

                # Combined score: (Confluence score is out of 10, explosive score is out of 100)
                combined_score = (confluence_score *
                                  10) + explosive_score  # Max 200

                final_opportunities.append({
                    'symbol':
                    symbol,
                    'combined_score':
                    round(combined_score, 2),
                    'explosive_score':
                    round(explosive_score, 2),
                    'confluence_score':
                    round(confluence_score, 2),
                    'confluence_bias':
                    core_data.get('confluence', {}).get('bias', 'N/A'),
                    'trade_plan':
                    core_data.get('trade_plan', {}),
                    'full_explosive_analysis':
                    explosive_data,
                    'full_core_analysis':
                    core_data
                })

        final_opportunities.sort(key=lambda x: x['combined_score'],
                                 reverse=True)

        print(
            f"\n🏆 FINAL RESULTS: Found {len(final_opportunities)} high-quality, two-phase opportunities."
        )

        response = {
            "status": "success",
            "scan_metadata": {
                "methodology":
                "Two-phase analysis combining options activity and technical confluence."
            },
            "final_opportunities": final_opportunities,
        }

        return jsonify(make_json_safe(response))

    except Exception as e:
        import traceback
        print(f"❌ FATAL ERROR in explosive-earnings-combo: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route("/pre-earnings-scan", methods=["GET", "POST"])
def run_pre_earnings_scan():
    """Run specialized scan focused on pre-earnings opportunities"""
    try:
        # Get parameters
        if request.method == 'POST' and request.is_json:
            data = request.get_json()
            days_ahead = int(data.get('days_ahead', 21))  # Look 21 days ahead
            min_delta = float(data.get('min_delta', 0.25))
            max_delta = float(data.get('max_delta', 0.68))
            priority_only = data.get('priority_only',
                                     True)  # Only critical/high priority
        else:
            days_ahead = int(request.args.get('days_ahead', 21))
            min_delta = float(request.args.get('min_delta', 0.25))
            max_delta = float(request.args.get('max_delta', 0.68))
            priority_only = request.args.get('priority_only',
                                             'true').lower() == 'true'

        from enhanced_scanner import discover_pre_earnings_stocks, run_enhanced_scanner

        # Get pre-earnings candidates with verbose output
        pre_earnings_stocks = discover_pre_earnings_stocks(verbose=True)

        if not pre_earnings_stocks:
            return jsonify({
                "status": "no-results",
                "message": "No stocks found with upcoming earnings"
            })

        # Run enhanced scanner on just these stocks
        results = run_enhanced_scanner(symbols=pre_earnings_stocks,
                                       min_delta=min_delta,
                                       max_delta=max_delta)

        # Filter by earnings priority if requested
        if priority_only and results:
            priority_results = {}
            for symbol, data in results.items():
                earnings_info = data.get('market_data',
                                         {}).get('earnings_info', {})
                priority = earnings_info.get('earnings_priority', 'low')
                if priority in ['critical', 'high']:
                    priority_results[symbol] = data
            results = priority_results

        if not results:
            return jsonify({
                "status": "no-results",
                "message": "No high-priority pre-earnings opportunities found",
                "candidates_scanned": len(pre_earnings_stocks)
            })

        # Calculate earnings-specific stats
        earnings_stats = {}
        for symbol, data in results.items():
            earnings_info = data.get('market_data',
                                     {}).get('earnings_info', {})
            priority = earnings_info.get('earnings_priority', 'unknown')
            days_to_earnings = earnings_info.get('days_to_earnings', 999)

            if priority not in earnings_stats:
                earnings_stats[priority] = []
            earnings_stats[priority].append({
                'symbol':
                symbol,
                'days_to_earnings':
                days_to_earnings,
                'confluence_score':
                data.get('confluence', {}).get('score', 0),
                "confidence":
                data.get('trade_plan', {}).get('validation_score', 0)
            })

        return jsonify({
            "status":
            "completed",
            "scan_type":
            "pre_earnings",
            "message":
            f"Pre-earnings scan completed successfully",
            "candidates_scanned":
            len(pre_earnings_stocks),
            "opportunities_found":
            len(results),
            "priority_filter":
            priority_only,
            "earnings_breakdown": {
                priority: len(stocks)
                for priority, stocks in earnings_stats.items()
            },
            "top_earnings_plays": [{
                "symbol":
                symbol,
                "days_to_earnings":
                data.get('market_data',
                         {}).get('earnings_info',
                                 {}).get('days_to_earnings', 999),
                "earnings_priority":
                data.get('market_data',
                         {}).get('earnings_info',
                                 {}).get('earnings_priority', 'unknown'),
                "confluence_score":
                data.get('confluence', {}).get('score', 0),
                "confidence":
                data.get('trade_plan', {}).get('validation_score', 0)
            } for symbol, data in list(results.items())[:10]]
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Pre-earnings scan failed: {str(e)}"
        }), 500


@app.route("/plans/formatted", methods=["GET"])
def get_formatted_plans():
    """Return human-readable formatted trading plans"""
    try:
        # Get query parameters
        sort_by = request.args.get('sort_by', 'confluence_score')
        order = request.args.get('order', 'desc')
        date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
        limit = int(request.args.get('limit', 10))  # Limit number of results

        # Get the sorted plans data
        base_dir = './TradingPlans'
        json_pattern = f'progressive_results_{date}.json'
        json_file = os.path.join(base_dir, json_pattern)

        if not os.path.exists(json_file):
            pattern = os.path.join(base_dir, 'progressive_results_*.json')
            files = glob.glob(pattern)
            if not files:
                return "No trading plans found", 404
            json_file = max(files, key=os.path.getctime)

        with open(json_file, 'r') as f:
            results = json.load(f)

        # Sort and format the results
        sorted_symbols = []
        for symbol, data in results.items():
            confluence_score = data.get('confluence', {}).get('score', 0)
            sorted_symbols.append((symbol, confluence_score, data))

        reverse_order = order.lower() == 'desc'
        if sort_by == 'confluence_score':
            sorted_symbols.sort(key=lambda x: x[1], reverse=reverse_order)
        else:
            sorted_symbols.sort(key=lambda x: x[0], reverse=reverse_order)

        # Limit results
        sorted_symbols = sorted_symbols[:limit]

        # Format as human-readable text
        formatted_output = f"Trading Plans - Sorted by {sort_by} ({order})\n"
        formatted_output += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        formatted_output += f"Total Plans: {len(sorted_symbols)}\n"
        formatted_output += "=" * 80 + "\n\n"

        for i, (symbol, score, data) in enumerate(sorted_symbols, 1):
            formatted_output += f"{i}. {symbol} - Confluence Score: {score:.1f}/10\n"
            formatted_output += f"   Bias: {data.get('confluence', {}).get('bias', 'N/A')}\n"

            if 'trade_plan' in data and data['trade_plan']:
                tp = data['trade_plan']
                formatted_output += f"   Entry: ${tp.get('entry_price', 0):.2f}\n"
                formatted_output += f"   Target: ${tp.get('initial_target', 0):.2f}\n"
                formatted_output += f"   Strike: {tp.get('strike', 'N/A')} {tp.get('type', 'N/A').capitalize()}\n"
                formatted_output += f"   Expiration: {tp.get('expiration', 'N/A')}\n"
                formatted_output += f"   Position Size: {tp.get('position_size', 0)} contracts\n"
                formatted_output += f"   Max Hold: {tp.get('max_hold_time', 'N/A')}\n"

            formatted_output += "\n" + "-" * 60 + "\n\n"

        return formatted_output, 200, {
            'Content-Type': 'text/plain; charset=utf-8'
        }

    except Exception as e:
        return f"Error retrieving formatted plans: {str(e)}", 500


@app.route("/enhanced-scan", methods=["GET", "POST"])
def run_enhanced_scan():
    """Run enhanced options scan with comprehensive analysis"""
    try:
        from enhanced_scanner import run_enhanced_scanner

        # Get parameters
        if request.method == 'POST' and request.is_json:
            data = request.get_json()
            symbols = data.get('symbols', None)
            min_delta = float(data.get('min_delta', 0.25))
            max_delta = float(data.get('max_delta', 0.68))
        else:
            symbols = request.args.getlist('symbols') or None
            min_delta = float(request.args.get('min_delta', 0.25))
            max_delta = float(request.args.get('max_delta', 0.68))

        # Run enhanced scanner
        results = run_enhanced_scanner(symbols=symbols,
                                       min_delta=min_delta,
                                       max_delta=max_delta)

        if not results:
            return jsonify({
                "status":
                "no-results",
                "message":
                "Enhanced scan completed but found no high-quality opportunities"
            })

        # Track performance for all results
        tracked_count = 0
        from performance_tracker import PerformanceTracker
        tracker = PerformanceTracker()

        for symbol, data in results.items():
            try:
                if ('options' in data
                        and not isinstance(data['options'], bool)
                        and not data['options'].empty and 'trade_plan' in data
                        and data['trade_plan']):

                    top_option = data['options'].iloc[0].to_dict()
                    trade_plan = data['trade_plan']

                    track_id = tracker.track_option_performance(
                        symbol, top_option, trade_plan,
                        datetime.now().strftime('%Y-%m-%d'))
                    tracked_count += 1
            except Exception as e:
                print(f"⚠️ Failed to track {symbol}: {e}")

        return jsonify({
            "status":
            "success",
            "scan_type":
            "enhanced_comprehensive",
            "opportunities_found":
            len(results),
            "top_opportunities": [{
                "symbol":
                symbol,
                "confluence_score":
                data.get('confluence', {}).get('score', 0),
                "bias":
                data.get('confluence', {}).get('bias', 'N/A'),
                "validation_score":
                data.get('trade_plan', {}).get('validation_score', 0)
            } for symbol, data in list(results.items())[:10]],
            "performance_tracking":
            f"Now tracking {tracked_count} options",
            "timestamp":
            datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Enhanced scan failed: {str(e)}"
        }), 500


@app.route("/enhanced-scan/progress", methods=["GET"])
def get_enhanced_scan_progress():
    """Check progress of enhanced scan"""
    try:
        from datetime import datetime
        import glob

        date_str = request.args.get('date',
                                    datetime.now().strftime('%Y-%m-%d'))
        progress_file = f'./TradingPlans/enhanced_scan_progress_{date_str}.json'

        if not os.path.exists(progress_file):
            return jsonify({
                "status":
                "not_found",
                "message":
                "No enhanced scan in progress for this date"
            })

        with open(progress_file, 'r') as f:
            progress_data = json.load(f)

        results = progress_data.get('results', {})
        processed_count = len(progress_data.get('processed_symbols', []))
        total_count = progress_data.get('total_symbols', 0)

        return jsonify({
            "status":
            "in_progress" if processed_count < total_count else "completed",
            "progress": {
                "processed":
                processed_count,
                "total":
                total_count,
                "percentage": (processed_count / total_count *
                               100) if total_count > 0 else 0,
                "remaining":
                total_count - processed_count
            },
            "results_found":
            len(results),
            "last_updated":
            progress_data.get('last_updated'),
            "top_opportunities": [{
                "symbol":
                symbol,
                "enhanced_score":
                data.get('confluence', {}).get('score', 0),
                "confidence":
                data.get('trade_plan', {}).get('validation_score', 0)
            } for symbol, data in list(results.items())[:5]]
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error checking progress: {str(e)}"
        }), 500


@app.route("/enhanced-scan/resume", methods=["POST"])
def resume_enhanced_scan():
    """Resume an interrupted enhanced scan"""
    try:
        from enhanced_scanner import run_enhanced_scanner

        # This will automatically detect and resume from existing progress
        results = run_enhanced_scanner()

        return jsonify({
            "status": "resumed",
            "message": "Enhanced scan resumed from last checkpoint",
            "opportunities_found": len(results) if results else 0
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to resume scan: {str(e)}"
        }), 500


@app.route("/explosive-techvol-scan", methods=["GET", "POST"])
def explosive_techvol_scan():
    """Find short-term technical or volatility driven setups."""
    try:
        from explosive_options_scanner import ExplosiveOptionsScanner

        if request.method == 'POST' and request.is_json:
            data = request.get_json()
            mode = data.get('mode', 'technical')
            top_n = int(data.get('top_n', 10))
            filters = data.get('filters', {})
        else:
            mode = request.args.get('mode', 'technical')
            top_n = int(request.args.get('top_n', 10))
            filters = {
                'min_price': float(request.args.get('min_price', 0.05)),
                'max_price': float(request.args.get('max_price', 5.00)),
                'min_delta': float(request.args.get('min_delta', 0.20)),
                'max_delta': float(request.args.get('max_delta', 0.40)),
                'min_days': int(request.args.get('min_days', 0)),
                'max_days': int(request.args.get('max_days', 7))
            }

        scanner = ExplosiveOptionsScanner(os.getenv('ALPHA_VANTAGE_API_KEY'))

        # Phase 1: get earnings plays to avoid duplicates
        earnings_results = scanner.run_explosive_scan(scan_type='earnings',
                                                      filters=filters)
        earnings_symbols = set(
            earnings_results.get('opportunities', {}).keys())

        # Phase 2: scan for unusual activity/technical setups
        results = scanner.run_explosive_scan(scan_type='unusual_activity',
                                             filters=filters)

        opportunities = []

        for symbol, data in results.get('opportunities', {}).items():
            if symbol in earnings_symbols:
                continue

            best = data.get('best_opportunity', {})
            market = data.get('market_data', {})

            dte = best.get('days_to_expiry', 0)
            delta = best.get('delta', 0)
            vol = best.get('volume', 0)
            mark = best.get('mark', 0)
            oi = best.get('open_interest', 1)
            iv = best.get('impliedVolatility', 0)
            hv = market.get('volatility_30d', 0) / 100
            price = market.get('current_price', 0)
            high = market.get('high', 0)
            iv_percentile = best.get('iv_percentile') or market.get(
                'iv_percentile')
            near_high = high > 0 and price >= high * 0.95
            volume_oi = vol / max(oi, 1)
            notional = vol * mark * 100

            low_iv = False
            if iv_percentile is not None:
                try:
                    ivp = float(iv_percentile)
                    low_iv = ivp <= 20
                except (TypeError, ValueError):
                    low_iv = False
            elif hv > 0:
                low_iv = iv < hv * 0.8

            earnings_flag = market.get('earnings_info',
                                       {}).get('is_pre_earnings', False)

            if (dte <= filters.get('max_days', 7) and 0.2 <= abs(delta) <= 0.4
                    and volume_oi >= 3 and low_iv and oi >= 100
                    and notional >= 1000):
                if mode == 'earnings' and not earnings_flag:
                    continue

                iv_score = 0
                if iv_percentile is not None:
                    iv_score = max(0, 20 - float(iv_percentile))
                elif hv:
                    iv_score = max(0, hv - iv) / hv * 20

                score = (min(volume_oi, 10) * 5 + iv_score +
                         (5 if near_high else 0) +
                         (5 if earnings_flag and mode == 'earnings' else 0))

                opportunities.append({
                    'symbol':
                    symbol,
                    'expiration':
                    best.get('expiration'),
                    'strike':
                    best.get('strike'),
                    'type':
                    best.get('type'),
                    'delta':
                    delta,
                    'theta':
                    best.get('theta'),
                    'gamma':
                    best.get('gamma'),
                    'iv':
                    iv,
                    'iv_percentile':
                    iv_percentile,
                    'volume':
                    vol,
                    'open_interest':
                    oi,
                    'dte':
                    dte,
                    'volume_oi_ratio':
                    round(volume_oi, 2),
                    'near_52w_high':
                    near_high,
                    'score':
                    round(score, 2),
                    'trading_plan':
                    data.get('trading_plan', {})
                })

        opportunities.sort(key=lambda x: x['score'], reverse=True)
        top_opps = opportunities[:top_n]

        summary = (f"Found {len(opportunities)} technical/vol opportunities. "
                   f"Top pick: {top_opps[0]['symbol'] if top_opps else 'N/A'}")

        return jsonify({
            'status': 'success',
            'mode': mode,
            'total_found': len(opportunities),
            'top_opportunities': top_opps,
            'summary': summary
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"Technical vol scan failed: {str(e)}"
        }), 500


@app.route("/explosive-52week-combo", methods=["GET", "POST"])
def explosive_52week_combo():
    """Two-phase scan: 52-week extremes discovery then full options analysis"""
    try:
        if request.method == 'POST' and request.is_json:
            data = request.get_json()
            min_explosive_score = data.get('min_explosive_score', 35)
            entry_price = data.get('entry_price')
            filters = data.get('filters', {})
            if entry_price and 'max_price' not in filters:
                filters['max_price'] = float(entry_price)
        else:
            min_explosive_score = float(
                request.args.get('min_explosive_score', 35))
            entry_price = request.args.get('entry_price')
            if entry_price:
                max_price_limit = float(entry_price)
            else:
                max_price_limit = float(request.args.get('max_price', 5.00))

            filters = {
                'min_price': float(request.args.get('min_price', 0.05)),
                'max_price': max_price_limit,
                'min_delta': float(request.args.get('min_delta', 0.10)),
                'max_delta': float(request.args.get('max_delta', 0.40)),
                'min_days': int(request.args.get('min_days', 1)),
                'max_days': int(request.args.get('max_days', 30))
            }

        print("🚀 EXPLOSIVE 52-WEEK COMBO SCAN - PHASE 1 + 2")
        print("=" * 70)
        print(f"🔥 Min Explosive Score: {min_explosive_score}")

        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')

        def track_call(state):
            now = time.time()
            if now - state['start'] >= 60:
                state['count'] = 0
                state['start'] = now
            state['count'] += 1
            if state['count'] >= 145:
                wait = 60 - (now - state['start']) + 1
                if wait > 0:
                    print(f"⏳ API throttle: waiting {wait:.1f}s")
                    time.sleep(wait)
                    state['count'] = 0
                    state['start'] = time.time()

        def fetch_price_history(symbol, state):
            track_call(state)
            url = (
                f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED'
                f'&symbol={symbol}&outputsize=full&apikey={api_key}')
            resp = requests.get(url, timeout=30)
            data = resp.json()
            if 'Time Series (Daily)' not in data:
                return None
            df = pd.DataFrame.from_dict(data['Time Series (Daily)'],
                                        orient='index')
            df.columns = [
                'Open', 'High', 'Low', 'Close', 'Adjusted_Close', 'Volume',
                'Dividend_Amount', 'Split_Coefficient'
            ]
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            return df.tail(260)

        from scanner_core import auto_discover_optionable_universe
        # Auto-discover an optionable symbol universe (deduped, prioritized)
        limit = int(request.args.get("limit", os.getenv("GAMMA_DISCOVERY_LIMIT", 300)))
        include_etfs = (request.args.get("include_etfs", os.getenv("INCLUDE_ETFS", "1")) == "1")
        symbols_to_scan = auto_discover_optionable_universe(limit=limit, include_etfs=include_etfs)
        # Deduplicate one more time defensively
        symbols_to_scan = sorted(set([s.upper() for s in symbols_to_scan if s]))
        print(f"\n🔍 Auto-discovered {len(symbols_to_scan)} symbols to scan (limit={limit}, include_etfs={include_etfs})")
        phase_state = {'count': 0, 'start': time.time()}
        extreme_symbols = {}

        print(
            f"📊 PHASE 1: Checking {len(symbols_to_scan)} symbols for extremes..."
        )
        for sym in symbols_to_scan:
            try:
                hist = fetch_price_history(sym, phase_state)
                if hist is None or hist.empty:
                    print(f"⚠️ No data for {sym}")
                    continue

                # Calculate 52-week highs/lows (or available data range)
                high_52 = hist['High'].max()
                low_52 = hist['Low'].min()
                last = hist.iloc[-1]
                last_price = float(last['Close'])
                last_volume = float(
                    last['Volume']) if not pd.isna(last['Volume']) else 0

                # Use more recent volume average (5-day instead of 20)
                recent_volume = hist['Volume'].tail(5)
                avg_volume = float(
                    recent_volume.mean()) if not recent_volume.empty else 1

                # More generous proximity thresholds
                near_high = last_price >= 0.90 * high_52  # Changed from 0.95 to 0.90
                near_low = last_price <= 1.10 * low_52  # Changed from 1.05 to 1.10
                volume_surge = last_volume >= 1.2 * avg_volume  # Changed from 1.5 to 1.2

                print(
                    f"📈 {sym}: Price ${last_price:.2f}, High ${high_52:.2f} ({last_price/high_52:.1%}), Low ${low_52:.2f} ({last_price/low_52:.1%}), Vol {last_volume/avg_volume:.1f}x"
                )

                # Score based on proximity to extremes AND volume
                score = 0

                if near_high:
                    # Score higher the closer to 52-week high
                    proximity_score = (last_price /
                                       high_52) * 30  # Max 30 points
                    score += proximity_score
                    print(
                        f"  🔥 Near 52-week high: +{proximity_score:.1f} points"
                    )

                if near_low:
                    # Score higher the closer to 52-week low (oversold bounce potential)
                    proximity_score = (
                        1 -
                        (last_price / low_52 - 1) / 0.10) * 25  # Max 25 points
                    score += proximity_score
                    print(
                        f"  📉 Near 52-week low: +{proximity_score:.1f} points")

                # Volume score (more weight for volume surge)
                volume_ratio = last_volume / max(avg_volume, 1)
                volume_score = min(volume_ratio * 20,
                                   40)  # Max 40 points for volume
                score += volume_score
                print(
                    f"  📊 Volume surge ({volume_ratio:.1f}x): +{volume_score:.1f} points"
                )

                # Recent momentum bonus (if price moved >2% today)
                if len(hist) > 1:
                    prev_close = hist.iloc[-2]['Close']
                    daily_change = (last_price - prev_close) / prev_close
                    if abs(daily_change) > 0.02:  # >2% move
                        momentum_score = min(abs(daily_change) * 100,
                                             15)  # Max 15 points
                        score += momentum_score
                        print(
                            f"  ⚡ Daily momentum ({daily_change:.1%}): +{momentum_score:.1f} points"
                        )

                score = round(score, 1)
                print(f"  🎯 Total Score: {score:.1f}")

                # Store all symbols with basic info (lower threshold for inclusion)
                if score > 10:  # Very low threshold to see more candidates
                    extreme_symbols[sym] = {
                        'last_price': last_price,
                        'last_volume': int(last_volume),
                        'avg_volume': int(avg_volume),
                        'high_52': high_52,
                        'low_52': low_52,
                        'volume_ratio': round(volume_ratio, 2),
                        'near_high': near_high,
                        'near_low': near_low,
                        'score': score,
                        'proximity_to_high': round(
                            (last_price / high_52) * 100, 1),
                        'proximity_to_low': round((last_price / low_52) * 100,
                                                  1)
                    }
                    print(f"  ✅ {sym} added with score {score:.1f}")

            except Exception as e:
                print(f"❌ Error processing {sym}: {e}")
                continue

        # Show top candidates regardless of min_explosive_score for debugging
        if extreme_symbols:
            sorted_symbols = sorted(extreme_symbols.items(),
                                    key=lambda x: x[1]['score'],
                                    reverse=True)
            print(f"\n🏆 TOP 10 CANDIDATES BY SCORE:")
            for i, (sym, data) in enumerate(sorted_symbols[:10], 1):
                print(
                    f"  {i}. {sym}: {data['score']:.1f} (High: {data['proximity_to_high']:.1f}%, Low: {data['proximity_to_low']:.1f}%, Vol: {data['volume_ratio']:.1f}x)"
                )

        # Use a more reasonable threshold - either min_explosive_score OR top 20% of found symbols
        score_threshold = min_explosive_score
        if extreme_symbols:
            # If min_explosive_score is too high, use 80th percentile of found scores
            all_scores = [d['score'] for d in extreme_symbols.values()]
            percentile_80 = np.percentile(all_scores, 80) if all_scores else 0
            score_threshold = min(min_explosive_score, max(
                percentile_80,
                25))  # At least 25, but not more than min_explosive_score

        candidates = [
            s for s, d in extreme_symbols.items()
            if d['score'] >= score_threshold
        ]

        print(
            f"✅ PHASE 1 COMPLETE: {len(candidates)} symbols meet criteria (threshold: {score_threshold:.1f})"
        )
        if score_threshold != min_explosive_score:
            print(
                f"  📊 Used adaptive threshold {score_threshold:.1f} instead of {min_explosive_score:.1f}"
            )

        # Show selected candidates
        if candidates:
            print(f"🎯 Selected candidates: {candidates[:10]}")  # Show first 10

        scanner_core_results = {}
        if candidates:
            from scanner_core import run_scanner
            print(
                f"🔬 PHASE 2: Running scanner_core on {len(candidates)} symbols"
            )
            scanner_core_results = run_scanner(
                symbols=candidates,
                min_delta=filters.get('min_delta', 0.10),
                max_delta=filters.get('max_delta', 0.40),
                min_price=filters.get('min_price', 0.05),
                max_price=filters.get('max_price', 5.00),
                time_to_expiry_range=(filters.get('min_days', 1),
                                      filters.get('max_days', 30)))

        final_opportunities = []
        for symbol in candidates:
            if symbol not in scanner_core_results:
                continue
            sc_data = scanner_core_results[symbol]
            trade_plan = sc_data.get('trade_plan', {})
            final_opportunities.append({
                'symbol':
                symbol,
                'option':
                trade_plan.get(
                    'option_symbol',
                    f"{symbol} {trade_plan.get('strike', '')} {trade_plan.get('type', '')}"
                ),
                'explosive_score':
                extreme_symbols[symbol]['score'],
                'confluence_score':
                sc_data.get('confluence', {}).get('score', 0),
                'confluence_bias':
                sc_data.get('confluence', {}).get('bias', 'N/A'),
                'trade_plan':
                trade_plan,
                'explosive_analysis':
                extreme_symbols[symbol],
                'scanner_core_analysis':
                sc_data
            })

        final_opportunities.sort(
            key=lambda x: (x['explosive_score'] + x['confluence_score']),
            reverse=True)

        combined_results = {
            'status': 'success',
            'scan_metadata': {
                'timestamp': datetime.now().isoformat(),
                'scan_type': 'explosive-52week-combo-2phase',
                'phase_1_symbols_scanned': len(symbols_to_scan),
                'phase_1_opportunities': len(extreme_symbols),
                'phase_2_symbols_analyzed': len(scanner_core_results),
                'min_explosive_score': min_explosive_score,
                'filters': filters
            },
            'phase_1_results': extreme_symbols,
            'phase_2_results': scanner_core_results,
            'final_opportunities': final_opportunities
        }

        top = final_opportunities[0] if final_opportunities else None
        summary = (f"📊 PHASE 1 – Scanned {len(symbols_to_scan)} symbols, "
                   f"{len(extreme_symbols)} met criteria. "
                   f"🔬 PHASE 2 – {len(scanner_core_results)} analyzed, "
                   f"{len(final_opportunities)} opportunities found.")
        if top:
            summary += (
                f" 🏆 Top: {top['symbol']} {top['trade_plan'].get('strike','')} "
                f"{top['trade_plan'].get('type','')} - Explosive {top['explosive_score']:.1f}/100, "
                f"Confluence {top['confluence_score']:.1f}/10")

        combined_results['summary'] = summary
        print(summary)
        return jsonify(make_json_safe(combined_results))

    except Exception as e:
        print(f"❌ Error in explosive-52week-combo: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route("/mega-discovery-scan", methods=["GET", "POST"])
def mega_discovery_scan():
    """Ultimate auto-discovery scan combining ALL methods with full analysis"""
    try:
        # Get parameters
        if request.method == 'POST' and request.is_json:
            data = request.get_json()
            max_symbols = int(data.get('max_symbols', 200))
            run_analysis = data.get('run_analysis', True)
            filters = data.get('filters', {})
        else:
            max_symbols = int(request.args.get('max_symbols', 200))
            run_analysis = request.args.get('run_analysis',
                                            'true').lower() == 'true'
            filters = {
                'min_price': float(request.args.get('min_price', 0.05)),
                'max_price': float(request.args.get('max_price', 5.00)),
                'min_delta': float(request.args.get('min_delta', 0.15)),
                'max_delta': float(request.args.get('max_delta', 0.35)),
                'min_days': int(request.args.get('min_days', 1)),
                'max_days': int(request.args.get('max_days', 21))
            }

        print("🚀 MEGA DISCOVERY SCAN - COMBINING ALL AUTO-DISCOVERY METHODS")
        print("=" * 70)

        all_discovered_symbols = []
        discovery_sources = {}
        # Method 1: Alpha Vantage Screeners
        try:
            print("📊 Method 1: Alpha Vantage Screeners...")
            av_data = fetch_alphavantage_top_symbols()
            av_symbols = av_data['all_symbols']
            all_discovered_symbols.extend(av_symbols)
            discovery_sources['alpha_vantage'] = {
                'count': len(av_symbols),
                'symbols': av_symbols[:20],  # Sample
                'categories': {
                    'top_gainers': len(av_data['top_gainers']),
                    'topgainers': len(av_data['top_gainers']),
                    'top_losers': len(av_data['top_losers']),
                    'most_active': len(av_data['most_active'])
                }
            }
            print(f"✅ Alpha Vantage: {len(av_symbols)} symbols")
        except Exception as e:
            print(f"⚠️ Alpha Vantage failed: {e}")
            discovery_sources['alpha_vantage'] = {'error': str(e)}

        # Method 2: Database symbols
        try:
            print("💾 Method 2: Database symbols...")
            from db_client import fetch_tickers_from_db
            db_symbols = fetch_tickers_from_db()
            all_discovered_symbols.extend(db_symbols)
            discovery_sources['database'] = {
                'count': len(db_symbols),
                'symbols': db_symbols[:20]
            }
            print(f"✅ Database: {len(db_symbols)} symbols")
        except Exception as e:
            print(f"⚠️ Database failed: {e}")
            discovery_sources['database'] = {'error': str(e)}

        # Method 3: High-volume optionable stocks
        try:
            print("📈 Method 3: High-volume optionable stocks...")
            from scanner_core import get_optionable_stocks_with_volume
            volume_symbols = get_optionable_stocks_with_volume()
            all_discovered_symbols.extend(volume_symbols)
            discovery_sources['high_volume'] = {
                'count': len(volume_symbols),
                'symbols': volume_symbols[:20]
            }
            print(f"✅ High Volume: {len(volume_symbols)} symbols")
        except Exception as e:
            print(f"⚠️ High volume method failed: {e}")
            discovery_sources['high_volume'] = {'error': str(e)}

        # Method 4: Earnings candidates
        try:
            print("📅 Method 4: Earnings candidates...")
            from enhanced_scanner import discover_pre_earnings_stocks
            earnings_symbols = discover_pre_earnings_stocks(verbose=False)
            all_discovered_symbols.extend(earnings_symbols)
            discovery_sources['earnings'] = {
                'count': len(earnings_symbols),
                'symbols': earnings_symbols[:20]
            }
            print(f"✅ Earnings: {len(earnings_symbols)} symbols")
        except Exception as e:
            print(f"⚠️ Earnings discovery failed: {e}")
            discovery_sources['earnings'] = {'error': str(e)}

        # Remove duplicates while preserving order
        unique_symbols = list(dict.fromkeys(all_discovered_symbols))
        print(
            f"🎯 DISCOVERY COMPLETE: {len(unique_symbols)} unique symbols from {len(all_discovered_symbols)} total"
        )

        # Limit symbols for performance
        if len(unique_symbols) > max_symbols:
            unique_symbols = unique_symbols[:max_symbols]
            print(f"⚡ Limited to {max_symbols} symbols for performance")

        mega_results = {
            "status": "success",
            "scan_type": "mega_discovery_combined",
            "discovery_summary": {
                "total_discovered": len(all_discovered_symbols),
                "unique_symbols": len(unique_symbols),
                "symbols_analyzed": len(unique_symbols) if run_analysis else 0,
                "discovery_sources": discovery_sources,
                "limited_to": max_symbols
            },
            "symbols": unique_symbols
        }

        # Run full scanner_core analysis if requested
        if run_analysis and unique_symbols:
            print(
                f"🔬 Running full scanner_core analysis on {len(unique_symbols)} symbols..."
            )
            try:
                from scanner_core import run_scanner
                analysis_results = run_scanner(
                    symbols=unique_symbols,
                    min_delta=filters.get('min_delta', 0.15),
                    max_delta=filters.get('max_delta', 0.35),
                    min_price=filters.get('min_price', 0.05),
                    max_price=filters.get('max_price', 5.00),
                    time_to_expiry_range=(filters.get('min_days', 1),
                                          filters.get('max_days', 21)))

                if analysis_results:
                    # Track performance for all results
                    tracked_count = 0
                    from performance_tracker import PerformanceTracker
                    tracker = PerformanceTracker()

                    for symbol, data in analysis_results.items():
                        try:
                            if ('options' in data
                                    and not isinstance(data['options'], bool)
                                    and not data['options'].empty
                                    and 'trade_plan' in data
                                    and data['trade_plan']):

                                top_option = data['options'].iloc[0].to_dict()
                                trade_plan = data['trade_plan']
                                track_id = tracker.track_option_performance(
                                    symbol, top_option, trade_plan,
                                    datetime.now().strftime('%Y-%m-%d'))
                                tracked_count += 1
                        except Exception as e:
                            print(f"⚠️ Failed to track {symbol}: {e}")

                    mega_results["analysis_results"] = analysis_results
                    mega_results["opportunities_found"] = len(analysis_results)
                    mega_results[
                        "performance_tracking"] = f"Now tracking {tracked_count} options"

                    # Add top opportunities summary
                    top_opportunities = sorted(
                        [(symbol, data.get('confluence', {}).get('score', 0))
                         for symbol, data in analysis_results.items()],
                        key=lambda x: x[1],
                        reverse=True)[:10]

                    mega_results["top_opportunities"] = [{
                        "symbol":
                        symbol,
                        "confluence_score":
                        score,
                        "bias":
                        analysis_results[symbol].get('confluence',
                                                     {}).get('bias', 'N/A')
                    } for symbol, score in top_opportunities]
                    print(
                        f"✅ Analysis complete: {len(analysis_results)} opportunities found"
                    )
                else:
                    mega_results["analysis_results"] = {}
                    mega_results["opportunities_found"] = 0
                    print("⚠️ Analysis completed but no opportunities found")

            except Exception as e:
                mega_results["analysis_error"] = str(e)
                print(f"❌ Analysis failed: {e}")

        print("🏁 MEGA DISCOVERY SCAN COMPLETE!")
        return jsonify(make_json_safe(mega_results))

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Mega discovery scan failed: {str(e)}"
        }), 500


@app.route("/discovery/info", methods=["GET"])
def discovery_info():
    """Show what each auto-discovery method finds without running analysis"""
    try:
        discovery_breakdown = {}

        # Test each discovery method
        print("🔍 Testing all auto-discovery methods...")

        # Alpha Vantage
        try:
            av_data = fetch_alphavantage_top_symbols()
            discovery_breakdown['alpha_vantage'] = {
                'status': 'success',
                'total_symbols': len(av_data['all_symbols']),
                'breakdown': {
                    'top_gainers': len(av_data['top_gainers']),
                    'top_losers': len(av_data['top_losers']),
                    'most_active': len(av_data['most_active'])
                },
                'sample_symbols': av_data['all_symbols'][:10]
            }
        except Exception as e:
            discovery_breakdown['alpha_vantage'] = {
                'status': 'error',
                'message': str(e)
            }

        # Database
        try:
            from db_client import fetch_tickers_from_db
            db_symbols = fetch_tickers_from_db()
            discovery_breakdown['database'] = {
                'status': 'success',
                'total_symbols': len(db_symbols),
                'sample_symbols': db_symbols[:10]
            }
        except Exception as e:
            discovery_breakdown['database'] = {
                'status': 'error',
                'message': str(e)
            }

        # High volume
        try:
            from scanner_core import get_optionable_stocks_with_volume
            volume_symbols = get_optionable_stocks_with_volume()
            discovery_breakdown['highvolume'] = {
                'status': 'success',
                'total_symbols': len(volume_symbols),
                'sample_symbols': volume_symbols[:10]
            }
        except Exception as e:
            discovery_breakdown['high_volume'] = {
                'status': 'error',
                'message': str(e)
            }

        # Earnings
        try:
            from enhanced_scanner import discover_pre_earnings_stocks
            earnings_symbols = discover_pre_earnings_stocks(verbose=False)
            discovery_breakdown['earnings'] = {
                'status': 'success',
                'total_symbols': len(earnings_symbols),
                'sample_symbols': earnings_symbols[:10]
            }
        except Exception as e:
            discovery_breakdown['earnings'] = {
                'status': 'error',
                'message': str(e)
            }

        # Calculate totals
        total_discovered = 0
        working_methods = 0
        for method, data in discovery_breakdown.items():
            if data.get('status') == 'success':
                total_discovered += data.get('total_symbols', 0)
                working_methods += 1

        return jsonify({
            "status": "success",
            "discovery_methods": discovery_breakdown,
            "summary": {
                "working_methods": working_methods,
                "total_methods": len(discovery_breakdown),
                "total_symbols_discovered": total_discovered,
                "health_check": "All methods tested"
            }
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Discovery info failed: {str(e)}"
        }), 500


@app.route("/quantitative-squeeze-scan", methods=["GET"])
def quantitative_squeeze_scan():
    """
    A dedicated endpoint to find high-potential explosive moves.
    It filters stocks based on a Bollinger Band Squeeze and high Relative Volume,
    then runs the full multi-timeframe analysis to find a confluence of signals.
    """
    try:
        # Get parameters from the request
        min_rvol = float(request.args.get('min_rvol', 2.0))
        min_confluence = float(request.args.get('min_confluence', 7.0))
        max_symbols_to_check = int(request.args.get('limit', 100))

        print("🚀 QUANTITATIVE SQUEEZE & VOLUME SCAN INITIATED")
        print("=" * 60)
        print(
            f"PARAMETERS: Min RVOL={min_rvol}, Min Confluence Score={min_confluence}"
        )

        from scanner_core import get_optionable_stocks_with_volume, run_scanner

        # --- PHASE 1: Quantitative Discovery ---
        print(
            "\n📊 PHASE 1: Discovering symbols and filtering for quantitative signals..."
        )

        # Use a robust method to get a list of potential symbols
        symbols_to_check = get_optionable_stocks_with_volume()
        if max_symbols_to_check > 0:
            symbols_to_check = symbols_to_check[:max_symbols_to_check]

        phase_1_hits = []
        for symbol in symbols_to_check:
            try:
                # 1. Check for Volatility Squeeze
                daily_history = yf.Ticker(symbol).history(period="1y")
                if daily_history.empty:
                    continue

                in_squeeze = is_in_bollinger_squeeze(daily_history)
                if not in_squeeze:
                    continue  # Skip if not in a squeeze

                # 2. Check for Relative Volume
                rvol = calculate_relative_volume(symbol)
                if rvol < min_rvol:
                    continue  # Skip if volume isn't high enough

                # If both conditions are met, it's a Phase 1 hit
                print(
                    f"🔥 PHASE 1 HIT: {symbol} is in a squeeze with RVOL of {rvol:.2f}"
                )
                phase_1_hits.append(symbol)
                time.sleep(1)  # Small delay to avoid API issues

            except Exception as e:
                print(f"⚠️ Error in Phase 1 for {symbol}: {e}")
                continue

        print(
            f"\n✅ PHASE 1 COMPLETE: Found {len(phase_1_hits)} candidates with strong quantitative signals."
        )

        if not phase_1_hits:
            return jsonify({
                "status": "no-results",
                "message": "Phase 1 found no candidates."
            })

        # --- PHASE 2: Deep Technical Analysis ---
        print(
            f"\n🔬 PHASE 2: Running scanner_core analysis on {len(phase_1_hits)} candidates..."
        )

        # Use your existing, powerful scanner_core on the highly-filtered list
        scanner_core_results = run_scanner(symbols=phase_1_hits)

        final_opportunities = []
        for symbol, core_data in scanner_core_results.items():
            confluence_score = core_data.get('confluence', {}).get('score', 0)

            if confluence_score >= min_confluence:
                print(
                    f"🏆 FINAL HIT: {symbol} passed all checks with Confluence Score of {confluence_score:.2f}"
                )
                final_opportunities.append({
                    "symbol":
                    symbol,
                    "confluence_score":
                    confluence_score,
                    "confluence_bias":
                    core_data.get('confluence', {}).get('bias', 'N/A'),
                    "trade_plan":
                    core_data.get('trade_plan', {})
                })

        print(
            f"\n🏁 SCAN COMPLETE: Found {len(final_opportunities)} fully-qualified opportunities."
        )

        return jsonify({
            "status": "success",
            "scan_parameters": {
                "min_rvol": min_rvol,
                "min_confluence_score": min_confluence
            },
            "opportunities": final_opportunities
        })

    except Exception as e:
        import traceback
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500



@app.route("/comprehensive-pipeline-scan", methods=["GET"])
def comprehensive_pipeline_scan():
    """
    Finds high-potential opportunities by looking for two distinct setups:
    1. SQUEEZE SETUP: Stocks currently in a volatility squeeze (potential energy).
    2. BREAKOUT SETUP: Stocks with a massive relative volume spike today (kinetic energy).
    """
    try:
        # --- ADD max_price TO PARAMETERS ---
        max_price = float(request.args.get('max_price', 2.00)) # Default to $2.00
        min_rvol = float(request.args.get('min_rvol', 2.0))
        min_confluence = float(request.args.get('min_confluence', 7.5))
        max_candidates = int(request.args.get('limit', 50))
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

        if not api_key:
            return jsonify({"status": "error", "message": "ALPHA_VANTAGE_API_KEY not set."}), 500

        print("🚀 COMPREHENSIVE PIPELINE SCAN INITIATED (SQUEEZE + BREAKOUT)")
        print("=" * 60)

        from enhanced_scanner import discover_pre_earnings_stocks
        from scanner_core import run_scanner

        # --- STAGE 1: DISCOVERY ---
        print("\n📊 STAGE 1: Discovering symbols...")
        symbols_to_check = discover_pre_earnings_stocks(verbose=False)
        print(f"✅ Found {len(symbols_to_check)} earnings candidates to check.")

        # --- STAGE 2: QUANTITATIVE FILTERING (Squeeze OR Breakout) ---
        print("\n🔬 STAGE 2: Filtering for Squeeze OR High RVOL...")

        quant_approved_symbols = []

        for symbol in symbols_to_check:
            if len(quant_approved_symbols) >= max_candidates:
                print(f"🎯 Candidate limit of {max_candidates} reached.")
                break
            try:
                in_squeeze = is_in_bollinger_squeeze(symbol, api_key=api_key)
                if in_squeeze:
                    print(f"🔥 SQUEEZE SETUP: {symbol} passed squeeze filter.")
                    quant_approved_symbols.append(symbol)
                    continue 

                rvol = calculate_relative_volume(symbol, api_key=api_key)
                if rvol >= min_rvol:
                    print(f"🔥 BREAKOUT SETUP: {symbol} passed RVOL filter ({rvol:.2f}).")
                    quant_approved_symbols.append(symbol)

            except Exception as e:
                print(f"⚠️ Error in Stage 2 for {symbol}: {e}")
                continue

        quant_approved_symbols = list(dict.fromkeys(quant_approved_symbols))

        print(f"\n✅ STAGE 2 COMPLETE: Found {len(quant_approved_symbols)} candidates meeting quantitative criteria.")

        if not quant_approved_symbols:
            return jsonify({"status": "no-results", "message": "No symbols passed the quantitative filters."})

        # --- STAGE 3: PROFILE SCAN & OPTIONS GENERATION ---
        print(f"\n📈 STAGE 3: Running deep profile scan on {len(quant_approved_symbols)} candidates...")

        # --- PASS max_price TO run_scanner ---
        final_results = run_scanner(symbols=quant_approved_symbols, max_price=max_price)

        final_opportunities = []
        for symbol, data in final_results.items():
            if data.get('confluence', {}).get('score', 0) >= min_confluence:
                final_opportunities.append({
                    "symbol": symbol,
                    "confluence_score": data['confluence']['score'],
                    "bias": data['confluence']['bias'],
                    "trade_plan": data.get('trade_plan')
                })

        print(f"\n🏁 PIPELINE COMPLETE: Found {len(final_opportunities)} fully-qualified trade plans.")

        return jsonify({
            "status": "success",
            "opportunities": final_opportunities
        })

    except Exception as e:
        import traceback
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500
        
@app.route("/gamma-squeeze-scan", methods=["GET"])
def gamma_squeeze_scan():
    """
    An innovative scanner that finds explosive setups by identifying "gamma walls"
    in the options chain that can trigger a feedback loop of forced buying.
    """
    print("🚀 GAMMA SQUEEZE SCAN INITIATED")
    print("=" * 60)

    from scanner_core import auto_discover_optionable_universe, run_scanner
    from explosive_options_scanner import ExplosiveOptionsScanner # We use this for its robust options fetching

    # Helper function to find a gamma wall for a single symbol
    def find_gamma_wall(symbol: str, scanner: ExplosiveOptionsScanner) -> dict or None:
        try:
            print(f"🔍 Analyzing gamma exposure for {symbol}...")
            options_df = scanner._fetch_all_options(symbol)

            if options_df is None or options_df.empty or 'gamma' not in options_df.columns:
                return None

            # Get current stock price for context
            stock_price = yf.Ticker(symbol).history(period="1d")['Close'].iloc[-1]

            # --- Gamma Calculation ---
            # 1. Filter for out-of-the-money (OTM) calls expiring within 45 days
            calls = options_df[
                (options_df['type'] == 'call') &
                (options_df['strike'] > stock_price) &
                (options_df['days_to_expiry'] <= 45)
            ].copy()

            if calls.empty: return None

            # 2. Calculate Total Gamma Exposure for each strike price
            # Total GEX = Gamma * Open Interest * 100 shares/contract
            calls['total_gamma'] = calls['gamma'] * calls['open_interest'] * 100

            # 3. Find the strike with the maximum gamma exposure (the "gamma wall")
            gamma_wall_strike = calls.loc[calls['total_gamma'].idxmax()]

            max_gamma = gamma_wall_strike['total_gamma']
            wall_strike_price = gamma_wall_strike['strike']

            # --- The Trigger Condition ---
            # Is the wall significant and within reach?
            distance_to_wall_pct = (wall_strike_price / stock_price - 1) * 100

            # Condition: Wall must be within 5-20% of the current price and have significant GEX
            if (5 < distance_to_wall_pct < 20) and (max_gamma > 10000): # 10,000 is a baseline for significant GEX
                print(f"  🔥 GAMMA WALL DETECTED for {symbol} at ${wall_strike_price:.2f}!")
                print(f"     - Stock is at ${stock_price:.2f} ({distance_to_wall_pct:.1f}% away)")
                print(f"     - Total Gamma Exposure: {max_gamma:,.0f}")

                return {
                    "symbol": symbol,
                    "wall_strike": wall_strike_price,
                    "distance_pct": distance_to_wall_pct,
                    "total_gamma": max_gamma,
                    "trigger_option": gamma_wall_strike.to_dict()
                }

        except Exception as e:
            print(f"  - Error analyzing gamma for {symbol}: {e}")
        return None

    # --- Main Scan Logic ---
    scanner = ExplosiveOptionsScanner(os.getenv('ALPHA_VANTAGE_API_KEY'))
    # Auto-discover an optionable symbol universe (deduped, prioritized)
    limit = int(request.args.get("limit", os.getenv("GAMMA_DISCOVERY_LIMIT", 300)))
    include_etfs = (request.args.get("include_etfs", os.getenv("INCLUDE_ETFS", "1")) == "1")
    symbols_to_check = auto_discover_optionable_universe(limit=limit, include_etfs=include_etfs)
    # Deduplicate one more time defensively
    symbols_to_check = sorted(set([s.upper() for s in symbols_to_check if s]))
    print(f"\n🔍 Auto-discovered {len(symbols_to_check)} symbols to scan (limit={limit}, include_etfs={include_etfs})")



    gamma_squeeze_candidates = []
    for symbol in symbols_to_check:
        wall_info = find_gamma_wall(symbol, scanner)
        if wall_info:
            gamma_squeeze_candidates.append(symbol)
        time.sleep(1) # API delay

    if not gamma_squeeze_candidates:
        return jsonify({"status": "no-results", "message": "No gamma squeeze setups found."})

    # --- Final Analysis on Filtered Candidates ---
    print(f"\n📈 Running final analysis on {len(gamma_squeeze_candidates)} gamma squeeze candidates...")
    final_results = run_scanner(symbols=gamma_squeeze_candidates)

    return jsonify({
        "status": "success",
        "scan_type": "gamma_squeeze",
        "opportunities": make_json_safe(final_results)
    })

if __name__ == "__main__":
    print("Starting Flask app on 0.0.0.0:8080...")
    try:
        app.run(host="0.0.0.0", port=8080, debug=True)
    except Exception as e:
        print(f"Error starting Flask app: {e}")
        raise
