from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageAnalysisSerializer
from .models import ImageAnalysis
from .predictions_engine import predict
import json

class AnalysisView(APIView):
    def post(self, request):
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Consistent Result Logic: Check if image hash already exists
        temp_instance = ImageAnalysis(image=image_file)
        temp_instance.save_hash_only()
        
        existing_analysis = ImageAnalysis.objects.filter(image_hash=temp_instance.image_hash).first()
        if existing_analysis:
            return Response(ImageAnalysisSerializer(existing_analysis).data, status=status.HTTP_200_OK)

        # Perform REAL AI Inference
        # Save image temporarily to disk for processing
        instance = ImageAnalysis(
            image=image_file,
            image_name=request.data.get('image_name', image_file.name),
            user_name=request.data.get('user_name', ''),
            user_type=request.data.get('user_type', '')
        )
        instance.save() # Saves file and hash

        try:
            img_path = instance.image.path
            
            # 1. Predict Bone Part
            bone_type = predict(img_path, model="Parts")
            
            # 2. Predict Fracture
            fracture_result = predict(img_path, model=bone_type)
            
            # Update instance with real results
            instance.bone_type = bone_type
            if isinstance(fracture_result, dict):
                instance.fracture_detected = (fracture_result['result'] == "DETECTED")
                instance.confidence = fracture_result.get('probability', 0.0) * 100
                instance.severity = "High Risk" if instance.fracture_detected else "Normal"
                instance.location = fracture_result.get('location', 'Unknown')
                instance.report_data = fracture_result
            
            instance.save()
            return Response(ImageAnalysisSerializer(instance).data, status=status.HTTP_201_CREATED)

        except Exception as e:
            return Response({"error": f"Inference failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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