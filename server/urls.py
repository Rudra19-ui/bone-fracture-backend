from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

from django.http import JsonResponse

def home(request):
    return JsonResponse({"status": "live", "message": "Bone Fracture Detection API is running"})

urlpatterns = [
    path('', home),
    path('admin/', admin.site.urls),
    path('api/', include('fracture.urls')),
    path('api/chatbot/', include('chatbot.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)