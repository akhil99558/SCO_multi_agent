from django.db import models

class AgentTask(models.Model):
    TASK_TYPES = (
        ('FORECAST', 'Demand Forecasting'),
        ('INVENTORY', 'Inventory Management'),
        ('PRICING', 'Pricing Optimization'),
        ('SUPPLY', 'Supply Chain Coordination'),
        ('CUSTOMER', 'Customer Behavior Analysis'),
    )
    
    STATUS_CHOICES = (
        ('PENDING', 'Pending'),
        ('RUNNING', 'Running'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    )
    
    task_type = models.CharField(max_length=50, choices=TASK_TYPES)
    status = models.CharField(max_length=50, choices=STATUS_CHOICES, default='PENDING')
    parameters = models.JSONField(default=dict)
    result = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    priority = models.IntegerField(default=1)
    
    def __str__(self):
        return f"{self.task_type} - {self.status}"
    
class AgentAction(models.Model):
    ACTION_TYPES = (
        ('REORDER', 'Place Reorder'),
        ('PRICE_CHANGE', 'Change Price'),
        ('TRANSFER', 'Transfer Inventory'),
        ('PROMOTION', 'Create Promotion'),
    )
    
    STATUS_CHOICES = (
        ('SUGGESTED', 'Suggested'),
        ('APPROVED', 'Approved'),
        ('REJECTED', 'Rejected'),
        ('EXECUTED', 'Executed'),
    )
    
    task = models.ForeignKey(AgentTask, on_delete=models.CASCADE, related_name='actions')
    action_type = models.CharField(max_length=50, choices=ACTION_TYPES)
    status = models.CharField(max_length=50, choices=STATUS_CHOICES, default='SUGGESTED')
    details = models.JSONField(default=dict)
    reasoning = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    executed_at = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.action_type} - {self.status}"    