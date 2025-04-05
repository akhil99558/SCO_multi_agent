import logging
from datetime import datetime
from .models import AgentTask, AgentAction
from django.db import transaction

logger = logging.getLogger(__name__)

class AgentCoordinator:
    """Central coordinator for multi-agent system"""
    
    def __init__(self):
        # Import agents here to avoid circular imports
        from agents.demand_agent import DemandForecastingAgent
        from agents.inventory_agent import InventoryManagementAgent
        from agents.pricing_agent import PricingOptimizationAgent
        from agents.supply_agent import SupplyChainAgent
        from agents.customer_agent import CustomerBehaviorAgent
        
        # Initialize agent instances
        self.agents = {
            'FORECAST': DemandForecastingAgent(),
            'INVENTORY': InventoryManagementAgent(),
            'PRICING': PricingOptimizationAgent(),
            'SUPPLY': SupplyChainAgent(),
            'CUSTOMER': CustomerBehaviorAgent(),
        }
    
    def create_task(self, task_type, parameters, priority=1):
        """Create a new task for an agent"""
        if task_type not in self.agents:
            raise ValueError(f"Unknown agent type: {task_type}")
            
        task = AgentTask.objects.create(
            task_type=task_type,
            parameters=parameters,
            priority=priority
        )
        
        # Queue task for processing
        self.process_task(task.id)
        return task
    
    def process_task(self, task_id):
        """Process a specific task"""
        try:
            task = AgentTask.objects.select_for_update().get(id=task_id)
            
            # Mark as running
            task.status = 'RUNNING'
            task.save()
            
            # Get the appropriate agent
            agent = self.agents[task.task_type]
            
            # Execute the agent's logic
            result = agent.process(task.parameters)
            
            # Update task with results
            task.result = result
            task.status = 'COMPLETED'
            task.save()
            
            # Generate actions based on results
            self._generate_actions(task, result)
            
            return task
            
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {str(e)}")
            task.status = 'FAILED'
            task.result = {'error': str(e)}
            task.save()
            raise
    
    @transaction.atomic
    def _generate_actions(self, task, result):
        """Generate actions based on agent results"""
        if 'recommended_actions' not in result:
            return
            
        for action_data in result['recommended_actions']:
            action = AgentAction.objects.create(
                task=task,
                action_type=action_data['action_type'],
                details=action_data['details'],
                reasoning=action_data.get('reasoning', 'No reasoning provided')
            )
            
        return True
    
    def approve_action(self, action_id):
        """Approve an action suggested by an agent"""
        action = AgentAction.objects.get(id=action_id)
        action.status = 'APPROVED'
        action.save()
        
        # Execute the action
        self._execute_action(action)
        
        return action
    
    def reject_action(self, action_id, reason):
        """Reject an action suggested by an agent"""
        action = AgentAction.objects.get(id=action_id)
        action.status = 'REJECTED'
        action.details['rejection_reason'] = reason
        action.save()
        return action
    
    def _execute_action(self, action):
        """Execute an approved action"""
        # Get agent that can execute this action
        agent = self.agents[action.task.task_type]
        
        # Execute action
        result = agent.execute_action(action.action_type, action.details)
        
        # Update action status
        action.status = 'EXECUTED'
        action.executed_at = datetime.now()
        action.details['execution_result'] = result
        action.save()
        
        return action
