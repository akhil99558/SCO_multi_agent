from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from agent_coordinator.models import AgentTask, AgentAction
from agent_coordinator.services import AgentCoordinator
from .serializers import AgentTaskSerializer, AgentActionSerializer

class AgentTaskViewSet(viewsets.ModelViewSet):
    queryset = AgentTask.objects.all()
    serializer_class = AgentTaskSerializer
    coordinator = AgentCoordinator()
    
    @action(detail=False, methods=['post'])
    def create_forecast_task(self, request):
        """API endpoint to create a demand forecasting task"""
        try:
            task = self.coordinator.create_task('FORECAST', request.data)
            return Response(
                self.get_serializer(task).data,
                status=status.HTTP_201_CREATED
            )
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
    
    @action(detail=True, methods=['post'])
    def process(self, request, pk=None):
        """Process a specific task"""
        try:
            task = self.coordinator.process_task(pk)
            return Response(self.get_serializer(task).data)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class AgentActionViewSet(viewsets.ModelViewSet):
    queryset = AgentAction.objects.all()
    serializer_class = AgentActionSerializer
    coordinator = AgentCoordinator()
    
    @action(detail=True, methods=['post'])
    def approve(self, request, pk=None):
        """Approve an action"""
        try:
            action = self.coordinator.approve_action(pk)
            return Response(self.get_serializer(action).data)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['post'])
    def reject(self, request, pk=None):
        """Reject an action"""
        try:
            reason = request.data.get('reason', 'No reason provided')
            action = self.coordinator.reject_action(pk, reason)
            return Response(self.get_serializer(action).data)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )