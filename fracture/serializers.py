from rest_framework import serializers
from .models import ImageAnalysis

class ImageAnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageAnalysis
        fields = '__all__'