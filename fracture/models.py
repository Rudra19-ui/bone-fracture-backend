from django.db import models
import hashlib

class ImageAnalysis(models.Model):
    image = models.ImageField(upload_to='uploads/')
    image_name = models.CharField(max_length=255)
    image_hash = models.CharField(max_length=64, blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    user_name = models.CharField(max_length=255, blank=True, null=True)
    user_type = models.CharField(max_length=100, blank=True, null=True)
    bone_type = models.CharField(max_length=100, blank=True, null=True)
    fracture_detected = models.BooleanField(default=False)
    confidence = models.FloatField(blank=True, null=True)
    severity = models.CharField(max_length=100, blank=True, null=True)
    location = models.CharField(max_length=255, blank=True, null=True)
    recommendations = models.JSONField(blank=True, null=True)
    risk_factors = models.JSONField(blank=True, null=True)
    treatment_plan = models.JSONField(blank=True, null=True)
    timeline = models.JSONField(blank=True, null=True)
    report_data = models.JSONField(blank=True, null=True)
    accuracy = models.FloatField(default=92.4) # Global model accuracy for reporting

    def save_hash_only(self):
        if self.image:
            try:
                h = hashlib.sha256()
                for chunk in self.image.chunks():
                    h.update(chunk)
                self.image_hash = h.hexdigest()
            except Exception:
                self.image_hash = None

    def save(self, *args, **kwargs):
        if self.image and not self.image_hash:
            self.save_hash_only()
        super().save(*args, **kwargs)