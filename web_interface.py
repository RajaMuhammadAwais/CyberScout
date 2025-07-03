
import os
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional
import threading
import uuid
import tempfile
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
from core.orchestrator import ReconOrchestrator
from core.output_manager import OutputManager
from utils.logger import setup_logger
from utils.validators import validate_target, get_target_type
from config import Config
from utils.deepseek import deepseek_complete

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'osint-recon-web-secret-key-change-in-production')

# Setup logging
logger = setup_logger('osint_web', verbose=True)

# Global storage for active reconnaissance tasks
active_tasks = {}
task_results = {}
# --- NEW: Completed tasks history (for /api/history endpoint) ---
completed_tasks_history = []



# --- DeepSeek AI-powered threat scoring endpoint ---
@app.route('/api/threat_score/<task_id>', methods=['GET'])
def get_task_threat_score(task_id):
    """Get AI-generated threat/risk score and justification for a completed task using DeepSeek."""
    try:
        task = active_tasks.get(task_id)
        if not task:
            return jsonify({'error': 'Task not found'}), 404
        if task.status != 'completed':
            return jsonify({'error': 'Task not completed yet'}), 400
        results = task_results.get(task_id)
        if not results:
            return jsonify({'error': 'Results not found'}), 404
        prompt = (
            f"Analyze the following OSINT reconnaissance results for target '{task.target}'. "
            f"Assign a risk/threat score from 1 (low) to 10 (critical) and provide a short justification.\nResults: {results}"
        )
        ai_response = deepseek_complete(prompt, max_tokens=256)
        return jsonify({'threat_score': ai_response})
    except Exception as e:
        logger.error(f"Failed to generate DeepSeek threat score: {e}")
        return jsonify({'error': str(e)}), 500

# --- DeepSeek AI-powered breach report summarization endpoint ---
@app.route('/api/summary/<task_id>', methods=['GET'])
def get_task_summary(task_id):
    """Get AI-generated summary of a completed reconnaissance task using DeepSeek."""
    try:
        task = active_tasks.get(task_id)
        if not task:
            return jsonify({'error': 'Task not found'}), 404
        if task.status != 'completed':
            return jsonify({'error': 'Task not completed yet'}), 400
        results = task_results.get(task_id)
        if not results:
            return jsonify({'error': 'Results not found'}), 404
        # Prepare a prompt for DeepSeek summarization
        prompt = f"Summarize the following OSINT reconnaissance results for target '{task.target}'. Highlight key findings, risks, and recommended actions.\nResults: {results}"
        summary = deepseek_complete(prompt, max_tokens=512)
        return jsonify({'summary': summary})
    except Exception as e:
        logger.error(f"Failed to generate DeepSeek summary: {e}")
        return jsonify({'error': str(e)}), 500

# --- DeepSeek AI enrichment endpoint ---
@app.route('/api/deepseek_enrich', methods=['POST'])
def deepseek_enrich():
    """AI-powered enrichment/summarization using DeepSeek."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip() if data else ''
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        summary = deepseek_complete(text)
        return jsonify({'result': summary})
    except Exception as e:
        logger.error(f"DeepSeek enrichment failed: {e}")
        return jsonify({'error': str(e)}), 500
#!/usr/bin/env python3
"""
Web Interface for OSINT Reconnaissance Tool
Flask-based web interface providing user-friendly access to reconnaissance capabilities
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import threading
import uuid

from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
import tempfile
from pathlib import Path

from core.orchestrator import ReconOrchestrator
from core.output_manager import OutputManager
from utils.logger import setup_logger
from utils.validators import validate_target, get_target_type
from config import Config

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'osint-recon-web-secret-key-change-in-production')

# Setup logging
logger = setup_logger('osint_web', verbose=True)


# Global storage for active reconnaissance tasks
active_tasks = {}
task_results = {}

# --- NEW: Completed tasks history (for /api/history endpoint) ---
completed_tasks_history = []

class ReconTask:
    """Represents an active reconnaissance task."""
    
    def __init__(self, task_id: str, target: str, modules: list, config: Config):
        self.task_id = task_id
        self.target = target
        self.modules = modules
        self.config = config
        self.status = 'pending'
        self.progress = 0
        self.results = None
        self.error = None
        self.start_time = None
        self.end_time = None
        self.thread = None
        self.current_module = None  # Track which module is running

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for JSON serialization."""
        return {
            'task_id': self.task_id,
            'target': self.target,
            'modules': self.modules,
            'status': self.status,
            'progress': self.progress,
            'current_module': self.current_module,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'error': self.error
        }

def run_reconnaissance_task(task: ReconTask):
    """Run reconnaissance task in a separate thread."""
    try:
        task.status = 'running'
        task.start_time = datetime.now()
        task.progress = 10
        logger.info(f"Starting reconnaissance task {task.task_id} for target: {task.target}")

        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Initialize orchestrator
            orchestrator = ReconOrchestrator(task.config)
            task.progress = 20
            logger.info(f"[Task {task.task_id}] Orchestrator initialized. Starting modules: {task.modules}")

            # --- Enhanced progress reporting ---
            async def run_with_progress():
                results = {}
                total = len(task.modules)
                for idx, module in enumerate(task.modules):
                    task.current_module = module
                    logger.info(f"[Task {task.task_id}] Starting module: {module} ({idx+1}/{total})")
                    try:
                        # Run the module via orchestrator (assume orchestrator has a run_single_module method)
                        if hasattr(orchestrator, 'run_single_module'):
                            mod_result = await orchestrator.run_single_module(task.target, module)
                        else:
                            # Fallback: run the whole set (legacy)
                            mod_result = await orchestrator.run_reconnaissance(task.target, [module])
                        results[module] = mod_result
                        logger.info(f"[Task {task.task_id}] Module {module} completed.")
                    except Exception as mod_e:
                        logger.error(f"[Task {task.task_id}] Module {module} failed: {mod_e}")
                        results[module] = {'error': str(mod_e)}
                    # Update progress after each module
                    task.progress = int(20 + 70 * (idx + 1) / total)  # 20-90%
                task.current_module = None
                return results

            # Run the enhanced async function
            results = loop.run_until_complete(run_with_progress())
            task.progress = 90
            task.results = results
            task.status = 'completed'
            task.progress = 100
            # Store results globally
            task_results[task.task_id] = results
            logger.info(f"Reconnaissance task {task.task_id} completed successfully")

        finally:
            loop.close()

    except Exception as e:
        task.status = 'failed'
        task.error = str(e)
        task.progress = 0
        logger.error(f"Reconnaissance task {task.task_id} failed: {e}")

    finally:
        task.end_time = datetime.now()

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/api/start_reconnaissance', methods=['POST'])
def start_reconnaissance():
    """Start a new reconnaissance task."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        target = data.get('target', '').strip()
        modules = data.get('modules', [])
        # Validate inputs
        if not target:
            return jsonify({'error': 'Target is required'}), 400
        if not validate_target(target):
            return jsonify({'error': 'Invalid target format'}), 400
        if not modules:
            return jsonify({'error': 'At least one module must be selected'}), 400
        # Create configuration
        config = Config(
            rate_limit=float(data.get('rate_limit', 1.0)),
            timeout=int(data.get('timeout', 30)),
            max_concurrent=int(data.get('max_concurrent', 10)),
            verbose=data.get('verbose', False)
        )
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        # Create task
        task = ReconTask(task_id, target, modules, config)
        active_tasks[task_id] = task
        # Start task in separate thread
        def run_and_record(task):
            run_reconnaissance_task(task)
            # If completed, add to history (keep last 100)
            if task.status == 'completed':
                completed_tasks_history.append(task.to_dict())
                if len(completed_tasks_history) > 100:
                    completed_tasks_history.pop(0)
        task.thread = threading.Thread(target=run_and_record, args=(task,))
        task.thread.daemon = True
        task.thread.start()
        logger.info(f"Started reconnaissance task {task_id}")
        return jsonify({
            'task_id': task_id,
            'status': 'started',
            'target': target,
            'modules': modules
        })
    except Exception as e:
        logger.error(f"Failed to start reconnaissance: {e}")
        return jsonify({'error': str(e)}), 500

# --- NEW: History endpoint for completed tasks ---
@app.route('/api/history')
def get_history():
    """Get history of completed tasks (last 100)."""
    try:
        # Optionally, allow filtering by target or module via query params
        target = request.args.get('target')
        module = request.args.get('module')
        limit = request.args.get('limit', default=20, type=int)
        offset = request.args.get('offset', default=0, type=int)
        export_format = request.args.get('format')  # 'json' or 'csv'
        filtered = completed_tasks_history
        if target:
            filtered = [t for t in filtered if t.get('target') == target]
        if module:
            filtered = [t for t in filtered if module in t.get('modules', [])]
        # Apply pagination
        total = len(filtered)
        paginated = filtered[offset:offset+limit]

        if export_format in ('json', 'csv'):
            # Export as file
            from flask import Response
            import csv
            import io
            filename = f"history_export.{export_format}"
            if export_format == 'json':
                content = json.dumps(paginated, indent=2)
                mimetype = 'application/json'
            else:
                # Convert list of dicts to CSV
                if not paginated:
                    content = ''
                else:
                    output = io.StringIO()
                    writer = csv.DictWriter(output, fieldnames=paginated[0].keys())
                    writer.writeheader()
                    writer.writerows(paginated)
                    content = output.getvalue()
                mimetype = 'text/csv'
            headers = {
                'Content-Disposition': f'attachment; filename={filename}'
            }
            return Response(content, mimetype=mimetype, headers=headers)

        # Default: return JSON API response
        return jsonify({'history': paginated, 'total': total, 'limit': limit, 'offset': offset})
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/task_status/<task_id>')
def get_task_status(task_id):
    """Get status of a reconnaissance task."""
    try:
        task = active_tasks.get(task_id)
        if not task:
            return jsonify({'error': 'Task not found'}), 404
        
        return jsonify(task.to_dict())
        
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/task_results/<task_id>')
def get_task_results(task_id):
    """Get results of a completed reconnaissance task."""
    try:
        task = active_tasks.get(task_id)
        if not task:
            return jsonify({'error': 'Task not found'}), 404
        
        if task.status != 'completed':
            return jsonify({'error': 'Task not completed yet'}), 400
        
        results = task_results.get(task_id)
        if not results:
            return jsonify({'error': 'Results not found'}), 404
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Failed to get task results: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/download_results/<task_id>/<format_type>')
def download_results(task_id, format_type):
    """Download results in specified format."""
    try:
        if format_type not in ['json', 'csv']:
            return jsonify({'error': 'Invalid format type'}), 400
        
        task = active_tasks.get(task_id)
        if not task:
            return jsonify({'error': 'Task not found'}), 404
        
        if task.status != 'completed':
            return jsonify({'error': 'Task not completed yet'}), 400
        
        results = task_results.get(task_id)
        if not results:
            return jsonify({'error': 'Results not found'}), 404
        
        # Generate output
        output_manager = OutputManager()
        
        if format_type == 'json':
            content = output_manager.format_json(results, task.target)
            filename = f"osint_results_{task.target}_{task_id[:8]}.json"
            mimetype = 'application/json'
        else:  # csv
            content = output_manager.format_csv(results, task.target)
            filename = f"osint_results_{task.target}_{task_id[:8]}.csv"
            mimetype = 'text/csv'
        
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, secure_filename(filename))
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return send_file(
            temp_file,
            as_attachment=True,
            download_name=filename,
            mimetype=mimetype
        )
        
    except Exception as e:
        logger.error(f"Failed to download results: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/validate_target', methods=['POST'])
def validate_target_endpoint():
    """Validate target format."""
    try:
        data = request.get_json()
        target = data.get('target', '').strip()
        
        if not target:
            return jsonify({'valid': False, 'error': 'Target is required'})
        
        is_valid = validate_target(target)
        target_type = get_target_type(target) if is_valid else 'unknown'
        
        return jsonify({
            'valid': is_valid,
            'target_type': target_type,
            'normalized_target': target.lower()
        })
        
    except Exception as e:
        logger.error(f"Target validation failed: {e}")
        return jsonify({'valid': False, 'error': str(e)})

@app.route('/api/active_tasks')
def get_active_tasks():
    """Get list of all active tasks."""
    try:
        tasks = []
        for task_id, task in active_tasks.items():
            tasks.append(task.to_dict())
        
        return jsonify({'tasks': tasks})
        
    except Exception as e:
        logger.error(f"Failed to get active tasks: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cancel_task/<task_id>', methods=['POST'])
def cancel_task(task_id):
    """Cancel a running task."""
    try:
        task = active_tasks.get(task_id)
        if not task:
            return jsonify({'error': 'Task not found'}), 404
        
        if task.status in ['completed', 'failed']:
            return jsonify({'error': 'Task already finished'}), 400
        
        # Note: In a production environment, you'd want more sophisticated
        # task cancellation. For now, we just mark it as cancelled.
        task.status = 'cancelled'
        task.end_time = datetime.now()
        
        logger.info(f"Task {task_id} cancelled")
        
        return jsonify({'status': 'cancelled'})
        
    except Exception as e:
        logger.error(f"Failed to cancel task: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system_info')
def get_system_info():
    """Get system information."""
    try:
        info = {
            'version': '1.0.0',
            'active_tasks': len([t for t in active_tasks.values() if t.status == 'running']),
            'total_tasks': len(active_tasks),
            'supported_modules': ['dns', 'dorks', 'breach', 'social', 'emails'],
            'supported_formats': ['json', 'csv']
        }
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    config = Config()
    print("Starting OSINT Reconnaissance Tool Web Interface...")
    print(f"Access at: http://localhost:{config.web_port}")
    app.run(host=config.web_host, port=config.web_port, debug=config.debug)
