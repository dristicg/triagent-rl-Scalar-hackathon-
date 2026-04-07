"""
TriageNet-RL Gradio Demo - Interactive Emergency Department Triage Simulation
Three tabs: Interactive Demo, Benchmark Results, and API Reference
"""

import gradio as gr
import json
import os
import pandas as pd
from typing import Dict, List, Optional, Tuple

from env import MedicalTriageEnv
from models import Action
from baseline import RuleBasedAgent


class TriageDemo:
    """Main demo application class."""
    
    def __init__(self):
        self.env = None
        self.agent = RuleBasedAgent()
        self.obs = None
        self.history = []
        self.episode_results = {}
    
    def format_patient_card(self, patient: Dict) -> str:
        """Format patient information as HTML card."""
        if not patient:
            return "<div style='color: #666;'>No current patient</div>"
        
        vitals = patient.get("vitals", {})
        vitals_html = f"""
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 4px; border: 1px solid #444;"><strong>BP</strong></td>
                <td style="padding: 4px; border: 1px solid #444;">{vitals.get('bp_systolic', '?')}/{vitals.get('bp_diastolic', '?')}</td>
                <td style="padding: 4px; border: 1px solid #444;"><strong>HR</strong></td>
                <td style="padding: 4px; border: 1px solid #444;">{vitals.get('heart_rate', '?')}</td>
            </tr>
            <tr>
                <td style="padding: 4px; border: 1px solid #444;"><strong>RR</strong></td>
                <td style="padding: 4px; border: 1px solid #444;">{vitals.get('respiratory_rate', '?')}</td>
                <td style="padding: 4px; border: 1px solid #444;"><strong>SpO2</strong></td>
                <td style="padding: 4px; border: 1px solid #444;">{vitals.get('spo2', '?')}%</td>
            </tr>
            <tr>
                <td style="padding: 4px; border: 1px solid #444;"><strong>Temp</strong></td>
                <td style="padding: 4px; border: 1px solid #444;">{vitals.get('temperature', '?')}°C</td>
                <td style="padding: 4px; border: 1px solid #444;"><strong>GCS</strong></td>
                <td style="padding: 4px; border: 1px solid #444;">{vitals.get('gcs', '?')}/15</td>
            </tr>
            <tr>
                <td style="padding: 4px; border: 1px solid #444;"><strong>Pain</strong></td>
                <td style="padding: 4px; border: 1px solid #444;">{vitals.get('pain_scale', '?')}/10</td>
                <td style="padding: 4px; border: 1px solid #444;"><strong>Wait</strong></td>
                <td style="padding: 4px; border: 1px solid #444;">{patient.get('wait_time_minutes', 0)} min</td>
            </tr>
        </table>
        """
        
        deteriorated_alert = ""
        if patient.get("deteriorated", False):
            deteriorated_alert = '<div style="background: #dc2626; color: white; padding: 8px; border-radius: 4px; margin: 8px 0;">🚨 PATIENT DETERIORATED</div>'
        
        return f"""
        <div style="background: #1e293b; color: white; padding: 16px; border-radius: 8px; margin: 8px 0;">
            <h3 style="margin: 0 0 12px 0;">{patient.get('name', 'Unknown')} ({patient.get('patient_id', 'Unknown')})</h3>
            <p style="margin: 4px 0;"><strong>Age:</strong> {patient.get('age', '?')} | <strong>Sex:</strong> {patient.get('gender', '?')}</p>
            <p style="margin: 4px 0;"><strong>Chief Complaint:</strong> {patient.get('chief_complaint', 'None')}</p>
            <p style="margin: 4px 0;"><strong>History:</strong> {', '.join(patient.get('medical_history', [])) or 'None'}</p>
            <p style="margin: 4px 0;"><strong>Meds:</strong> {', '.join(patient.get('current_medications', [])) or 'None'}</p>
            {deteriorated_alert}
            <h4 style="margin: 12px 0 4px 0;">Vitals:</h4>
            {vitals_html}
        </div>
        """
    
    def format_env_status(self, obs) -> str:
        """Format environment status as HTML."""
        if not obs:
            return "<div style='color: #666;'>Environment not initialized</div>"
        
        bed_status = obs.bed_status
        alerts_html = ""
        if obs.alerts:
            alerts_html = "<br>".join([f"🚨 {alert}" for alert in obs.alerts])
        else:
            alerts_html = "None"
        
        return f"""
        <div style="background: #f8fafc; padding: 16px; border-radius: 8px; margin: 8px 0;">
            <h4 style="margin: 0 0 8px 0;">Environment Status</h4>
            <p style="margin: 4px 0;"><strong>Step:</strong> {obs.step}/{obs.max_steps}</p>
            <p style="margin: 4px 0;"><strong>Time:</strong> {obs.time_elapsed_minutes} minutes elapsed</p>
            <p style="margin: 4px 0;"><strong>Beds Available:</strong> {bed_status.get('available_beds', 0)} total</p>
            <p style="margin: 4px 0;">&nbsp;&nbsp;&nbsp;Resuscitation: {bed_status.get('resuscitation_available', 0)}</p>
            <p style="margin: 4px 0;">&nbsp;&nbsp;&nbsp;Acute Care: {bed_status.get('acute_care_available', 0)}</p>
            <p style="margin: 4px 0;"><strong>Queue:</strong> {len(obs.pending_patients)} pending, {len(obs.triaged_patients)} triaged</p>
            <p style="margin: 4px 0;"><strong>Cumulative Score:</strong> {obs.score_so_far:+.3f}</p>
            <p style="margin: 4px 0;"><strong>Alerts:</strong> {alerts_html}</p>
        </div>
        """
    
    def format_history_table(self, history: List[Dict]) -> str:
        """Format step history as HTML table."""
        if not history:
            return "<div style='color: #666;'>No actions taken yet</div>"
        
        # Show last 15 steps
        recent_history = history[-15:]
        
        rows = []
        for i, step in enumerate(recent_history):
            rows.append(f"""
                <tr>
                    <td style="padding: 4px; border: 1px solid #ddd;">{step.get('step', '?')}</td>
                    <td style="padding: 4px; border: 1px solid #ddd;">{step.get('action', '?')}</td>
                    <td style="padding: 4px; border: 1px solid #ddd;">{step.get('patient', '?')}</td>
                    <td style="padding: 4px; border: 1px solid #ddd;">{step.get('esi', '-')}</td>
                    <td style="padding: 4px; border: 1px solid #ddd;">{step.get('reward', '?'):>+7.3f}</td>
                    <td style="padding: 4px; border: 1px solid #ddd;">{step.get('feedback', '')[:50]}...</td>
                </tr>
            """)
        
        return f"""
        <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
            <thead>
                <tr style="background: #f1f5f9;">
                    <th style="padding: 6px; border: 1px solid #ddd;">Step</th>
                    <th style="padding: 6px; border: 1px solid #ddd;">Action</th>
                    <th style="padding: 6px; border: 1px solid #ddd;">Patient</th>
                    <th style="padding: 6px; border: 1px solid #ddd;">ESI</th>
                    <th style="padding: 6px; border: 1px solid #ddd;">Reward</th>
                    <th style="padding: 6px; border: 1px solid #ddd;">Feedback</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """
    
    def start_episode(self, task_choice: str) -> Tuple[str, str, str]:
        """Start a new episode with selected task."""
        task_map = {"Task 1 (Easy, 1 patient)": 1, "Task 2 (Medium, 8 patients)": 2, "Task 3 (Hard, 15 patients)": 3}
        task_id = task_map[task_choice]
        
        self.env = MedicalTriageEnv(task_id=task_id, seed=42)
        self.obs = self.env.reset()
        self.history = []
        
        patient_html = self.format_patient_card(self.obs.current_patient)
        status_html = self.format_env_status(self.obs)
        history_html = self.format_history_table(self.history)
        
        return patient_html, status_html, history_html
    
    def step_agent(self) -> Tuple[str, str, str]:
        """Take one step with the baseline agent."""
        if self.env is None or self.obs is None:
            return "<div style='color: red;'>Please start an episode first</div>", "", ""
        
        # Get agent action
        action = self.agent.select_action(self.obs.model_dump())
        
        # Execute step
        result = self.env.step(action)
        self.obs = result.observation
        
        # Add to history
        self.history.append({
            'step': self.obs.step,
            'action': action.action_type,
            'patient': action.patient_id,
            'esi': action.esi_level or '-',
            'reward': result.reward.step_reward,
            'feedback': result.reward.feedback
        })
        
        patient_html = self.format_patient_card(self.obs.current_patient)
        status_html = self.format_env_status(self.obs)
        history_html = self.format_history_table(self.history)
        
        return patient_html, status_html, history_html
    
    def run_full_baseline(self, task_choice: str) -> str:
        """Run full baseline agent on selected task."""
        task_map = {"Task 1 (Easy, 1 patient)": 1, "Task 2 (Medium, 8 patients)": 2, "Task 3 (Hard, 15 patients)": 3}
        task_id = task_map[task_choice]
        
        env = MedicalTriageEnv(task_id=task_id, seed=42)
        agent = RuleBasedAgent()
        obs = env.reset()
        
        done = False
        steps = 0
        total_reward = 0.0
        
        while not done and steps < 100:
            action = agent.select_action(obs.model_dump())
            result = env.step(action)
            steps += 1
            total_reward = result.reward.cumulative_reward
            done = result.done
            obs = result.observation
        
        grade = env.grade()
        
        result_html = f"""
        <div style="background: #f0f9ff; padding: 16px; border-radius: 8px; margin: 8px 0;">
            <h3>Baseline Agent Results - Task {task_id}</h3>
            <p><strong>Final Score:</strong> {grade.final_score:.3f} / 1.000</p>
            <p><strong>Total Reward:</strong> {grade.total_reward:+.3f}</p>
            <p><strong>Steps Taken:</strong> {grade.steps_taken}</p>
            <p><strong>Correct ESI:</strong> {grade.correct_esi_count}/{grade.total_patients}</p>
            <p><strong>Critical Misses:</strong> {grade.critical_misses}</p>
            <p><strong>Deteriorations:</strong> {grade.deteriorations}</p>
            <p><strong>Passed:</strong> {'✅ YES' if grade.passed else '❌ NO'}</p>
            
            <h4>Grade Breakdown:</h4>
            <ul>
        """
        
        for k, v in grade.grade_breakdown.items():
            result_html += f"<li><strong>{k}:</strong> {v:+.3f}</li>"
        
        result_html += "</ul></div>"
        return result_html
    
    def get_benchmark_scores(self) -> str:
        """Get benchmark scores table."""
        scores_html = """
        <div style="background: #f8fafc; padding: 16px; border-radius: 8px;">
            <h3>Baseline Agent Benchmark Results</h3>
            <table style="width: 100%; border-collapse: collapse; margin: 12px 0;">
                <thead>
                    <tr style="background: #e2e8f0;">
                        <th style="padding: 8px; border: 1px solid #cbd5e1;">Task</th>
                        <th style="padding: 8px; border: 1px solid #cbd5e1;">Score</th>
                        <th style="padding: 8px; border: 1px solid #cbd5e1;">Difficulty</th>
                        <th style="padding: 8px; border: 1px solid #cbd5e1;">Status</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #cbd5e1;">Task 1</td>
                        <td style="padding: 8px; border: 1px solid #cbd5e1;">0.680</td>
                        <td style="padding: 8px; border: 1px solid #cbd5e1;">Easy</td>
                        <td style="padding: 8px; border: 1px solid #cbd5e1;">✅</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #cbd5e1;">Task 2</td>
                        <td style="padding: 8px; border: 1px solid #cbd5e1;">0.610</td>
                        <td style="padding: 8px; border: 1px solid #cbd5e1;">Medium</td>
                        <td style="padding: 8px; border: 1px solid #cbd5e1;">✅</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #cbd5e1;">Task 3</td>
                        <td style="padding: 8px; border: 1px solid #cbd5e1;">0.520</td>
                        <td style="padding: 8px; border: 1px solid #cbd5e1;">Hard</td>
                        <td style="padding: 8px; border: 1px solid #cbd5e1;">❌</td>
                    </tr>
                    <tr style="background: #fef3c7; font-weight: bold;">
                        <td style="padding: 8px; border: 1px solid #cbd5e1;">AVERAGE</td>
                        <td style="padding: 8px; border: 1px solid #cbd5e1;">0.603</td>
                        <td style="padding: 8px; border: 1px solid #cbd5e1;">-</td>
                        <td style="padding: 8px; border: 1px solid #cbd5e1;">-</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """
        return scores_html
    
    def get_esi_reference(self) -> str:
        """Get ESI reference table."""
        return """
        <div style="background: #fefce8; padding: 16px; border-radius: 8px;">
            <h3>Emergency Severity Index (ESI) Reference</h3>
            <table style="width: 100%; border-collapse: collapse; margin: 12px 0;">
                <thead>
                    <tr style="background: #fde047;">
                        <th style="padding: 8px; border: 1px solid #facc15;">ESI Level</th>
                        <th style="padding: 8px; border: 1px solid #facc15;">Description</th>
                        <th style="padding: 8px; border: 1px solid #facc15;">Examples</th>
                        <th style="padding: 8px; border: 1px solid #facc15;">Target Time</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #facc15; font-weight: bold;">1</td>
                        <td style="padding: 8px; border: 1px solid #facc15;">Immediate - Life Threat</td>
                        <td style="padding: 8px; border: 1px solid #facc15;">Cardiac arrest, unresponsive, active seizure</td>
                        <td style="padding: 8px; border: 1px solid #facc15;">Immediate</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #facc15; font-weight: bold;">2</td>
                        <td style="padding: 8px; border: 1px solid #facc15;">Emergent - High Risk</td>
                        <td style="padding: 8px; border: 1px solid #facc15;">Chest pain, stroke, severe dyspnea</td>
                        <td style="padding: 8px; border: 1px solid #facc15;">&lt; 10 min</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #facc15; font-weight: bold;">3</td>
                        <td style="padding: 8px; border: 1px solid #facc15;">Urgent - Stable</td>
                        <td style="padding: 8px; border: 1px solid #facc15;">Abdominal pain, moderate pain</td>
                        <td style="padding: 8px; border: 1px solid #facc15;">&lt; 1 hour</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #facc15; font-weight: bold;">4</td>
                        <td style="padding: 8px; border: 1px solid #facc15;">Less Urgent</td>
                        <td style="padding: 8px; border: 1px solid #facc15;">Mild sprain, sore throat</td>
                        <td style="padding: 8px; border: 1px solid #facc15;">&lt; 1 hour</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #facc15; font-weight: bold;">5</td>
                        <td style="padding: 8px; border: 1px solid #facc15;">Non-Urgent</td>
                        <td style="padding: 8px; border: 1px solid #facc15;">Prescription refill, minor cut</td>
                        <td style="padding: 8px; border: 1px solid #facc15;">&lt; 2 hours</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """
    
    def get_api_reference(self) -> str:
        """Get API reference documentation."""
        return """
        <div style="background: #f0fdf4; padding: 16px; border-radius: 8px;">
            <h3>API Reference</h3>
            
            <h4>Core Methods</h4>
            <div style="background: #f9fafb; padding: 12px; border-radius: 4px; margin: 8px 0;">
                <code>env = MedicalTriageEnv(task_id=1, seed=42)</code>
                <p>Create environment instance</p>
            </div>
            
            <div style="background: #f9fafb; padding: 12px; border-radius: 4px; margin: 8px 0;">
                <code>obs = env.reset()</code>
                <p>Reset environment and get initial observation</p>
            </div>
            
            <div style="background: #f9fafb; padding: 12px; border-radius: 4px; margin: 8px 0;">
                <code>result = env.step(action)</code>
                <p>Execute action and get result</p>
            </div>
            
            <div style="background: #f9fafb; padding: 12px; border-radius: 4px; margin: 8px 0;">
                <code>grade = env.grade()</code>
                <p>Grade completed episode</p>
            </div>
            
            <h4>Action Types</h4>
            <table style="width: 100%; border-collapse: collapse; margin: 12px 0;">
                <tr><th style="padding: 4px; border: 1px solid #d1d5db;">Action</th><th style="padding: 4px; border: 1px solid #d1d5db;">Description</th></tr>
                <tr><td style="padding: 4px; border: 1px solid #d1d5db;">assign_esi</td><td style="padding: 4px; border: 1px solid #d1d5db;">Assign ESI level and routing</td></tr>
                <tr><td style="padding: 4px; border: 1px solid #d1d5db;">request_resources</td><td style="padding: 4px; border: 1px solid #d1d5db;">Request clinical resources</td></tr>
                <tr><td style="padding: 4px; border: 1px solid #d1d5db;">escalate</td><td style="padding: 4px; border: 1px solid #d1d5db;">Escalate critical case</td></tr>
                <tr><td style="padding: 4px; border: 1px solid #d1d5db;">reassess</td><td style="padding: 4px; border: 1px solid #d1d5db;">Reassess patient</td></tr>
                <tr><td style="padding: 4px; border: 1px solid #d1d5db;">discharge</td><td style="padding: 4px; border: 1px solid #d1d5db;">Discharge patient</td></tr>
                <tr><td style="padding: 4px; border: 1px solid #d1d5db;">wait</td><td style="padding: 4px; border: 1px solid #d1d5db;">Wait/do nothing</td></tr>
            </table>
            
            <h4>Routing Zones</h4>
            <table style="width: 100%; border-collapse: collapse; margin: 12px 0;">
                <tr><th style="padding: 4px; border: 1px solid #d1d5db;">Zone</th><th style="padding: 4px; border: 1px solid #d1d5db;">ESI Levels</th></tr>
                <tr><td style="padding: 4px; border: 1px solid #d1d5db;">resuscitation_bay</td><td style="padding: 4px; border: 1px solid #d1d5db;">ESI 1</td></tr>
                <tr><td style="padding: 4px; border: 1px solid #d1d5db;">acute_care</td><td style="padding: 4px; border: 1px solid #d1d5db;">ESI 2</td></tr>
                <tr><td style="padding: 4px; border: 1px solid #d1d5db;">fast_track</td><td style="padding: 4px; border: 1px solid #d1d5db;">ESI 3</td></tr>
                <tr><td style="padding: 4px; border: 1px solid #d1d5db;">waiting_room</td><td style="padding: 4px; border: 1px solid #d1d5db;">ESI 4, 5</td></tr>
            </table>
        </div>
        """


def create_demo():
    """Create the Gradio demo interface."""
    demo = TriageDemo()
    
    with gr.Blocks(
        title="🏥 Medical Triage OpenEnv",
    ) as interface:
        gr.Markdown("# 🏥 TriageNet-RL - Emergency Department Triage Simulation")
        gr.Markdown("Interactive demo of AI-powered emergency department triage using the Emergency Severity Index (ESI) standard.")
        
        with gr.Tabs():
            # Tab 1: Interactive Demo
            with gr.TabItem("🎮 Interactive Demo"):
                with gr.Row():
                    with gr.Column(scale=1):
                        task_choice = gr.Radio(
                            choices=["Task 1 (Easy, 1 patient)", "Task 2 (Medium, 8 patients)", "Task 3 (Hard, 15 patients)"],
                            value="Task 1 (Easy, 1 patient)",
                            label="Select Task"
                        )
                        
                        with gr.Row():
                            start_btn = gr.Button("▶ Start Episode", variant="primary")
                            step_btn = gr.Button("⏭ Step Agent")
                            run_btn = gr.Button("🚀 Run Full Baseline", variant="secondary")
                    
                    with gr.Column(scale=2):
                        patient_display = gr.HTML("", label="Current Patient")
                
                with gr.Row():
                    with gr.Column():
                        env_status = gr.HTML("", label="Environment Status")
                    
                    with gr.Column():
                        results_display = gr.HTML("", label="Results")
                
                history_display = gr.HTML("", label="Step History")
                
                # Event handlers
                start_btn.click(
                    demo.start_episode,
                    inputs=[task_choice],
                    outputs=[patient_display, env_status, history_display]
                )
                
                step_btn.click(
                    demo.step_agent,
                    outputs=[patient_display, env_status, history_display]
                )
                
                run_btn.click(
                    demo.run_full_baseline,
                    inputs=[task_choice],
                    outputs=[results_display]
                )
            
            # Tab 2: Benchmark
            with gr.TabItem("📈 Benchmark"):
                gr.HTML(demo.get_benchmark_scores())
                gr.HTML(demo.get_esi_reference())
            
            # Tab 3: API Reference
            with gr.TabItem("📖 API Reference"):
                gr.HTML(demo.get_api_reference())
    
    return interface


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False,
    )
