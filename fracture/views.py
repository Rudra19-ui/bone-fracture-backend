from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageAnalysisSerializer
from .models import ImageAnalysis

class AnalysisView(APIView):
    def post(self, request):
        data = request.data.copy()
        
        # Consistent Result Logic: Check if image hash already exists
        temp_instance = ImageAnalysis(image=request.FILES.get('image'))
        temp_instance.save_hash_only() # We need a method to just compute hash without full save
        
        existing_analysis = ImageAnalysis.objects.filter(image_hash=temp_instance.image_hash).first()
        if existing_analysis:
            return Response(ImageAnalysisSerializer(existing_analysis).data, status=status.HTTP_200_OK)

        serializer = ImageAnalysisSerializer(data=data)
        if serializer.is_valid():
            instance = serializer.save()
            return Response(ImageAnalysisSerializer(instance).data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request):
        image_name = request.query_params.get('image_name')
        image_hash = request.query_params.get('image_hash')
        if image_hash:
            obj = ImageAnalysis.objects.filter(image_hash=image_hash).order_by('-uploaded_at').first()
            if obj:
                return Response(ImageAnalysisSerializer(obj).data)
            return Response({}, status=status.HTTP_200_OK)
        if image_name:
            obj = ImageAnalysis.objects.filter(image_name=image_name).order_by('-uploaded_at').first()
            if obj:
                return Response(ImageAnalysisSerializer(obj).data)
            return Response({}, status=status.HTTP_200_OK)
        qs = ImageAnalysis.objects.order_by('-uploaded_at')[:20]
        return Response(ImageAnalysisSerializer(qs, many=True).data)