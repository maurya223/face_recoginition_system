import base64
import json
import numpy as np
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.utils.dateparse import parse_date, parse_time
from .models import Attendance
from .face_recognition_engine import recognize_face

def home(request):
    return render(request, 'home.html')

def register(request):
    return render(request, 'register.html')

def login(request):
    return render(request, 'login.html')

def attendance(request):
    return render(request, 'attendance.html')

def get_attendance(request):
    attendances = list(Attendance.objects.all().values('name', 'date', 'time'))
    return JsonResponse({'attendances': attendances})

@csrf_exempt
def mark_attendance_manual(request):
    if request.method == 'POST':
        name = request.POST.get('name', '').strip()
        if name:
            Attendance.objects.create(name=name)
            attendances = list(Attendance.objects.all().values('name', 'date', 'time')[:50])  # Last 50
            return JsonResponse({'status': 'success', 'attendances': attendances})
    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)

@csrf_exempt
def mark_attendance(request):
    # Image-based attendance (stub - integrate camera)
    if request.method == 'POST':
        # TODO: Process image from request.FILES or base64
        name = 'Demo User'  # Replace with recognition
        Attendance.objects.create(name=name)
        attendances = list(Attendance.objects.all().values('name', 'date', 'time')[:50])
        return JsonResponse({'status': 'success', 'attendances': attendances})
    return JsonResponse({'status': 'error'}, status=400)

@csrf_exempt
def ajax_recognize_face(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            frame_b64 = data.get('frame')
            frame = base64.b64decode(frame_b64.split(',')[1])  # Remove data:image prefix
            import cv2
            frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
            name = recognize_face(frame)
            return JsonResponse({'name': name})
        except Exception as e:
            return JsonResponse({'name': 'Unknown', 'error': str(e)}, status=400)
    return JsonResponse({'error': 'POST required'}, status=400)

@csrf_exempt
def register_face(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            name = data.get('name', 'Unknown')
            images = data.get('images', [])
            if not images:
                return JsonResponse({'success': False, 'error': 'No images provided'})
            
            # Process first image
            frame_b64 = images[0]
            frame = base64.b64decode(frame_b64.split(',')[1])
            import cv2
            frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
            
            from app.recognition import get_face_embedding
            from app.detect_face import detect_faces
            from app.vector_db import add_user, load_db
            
            load_db()
            faces = detect_faces(frame)
            if not faces:
                return JsonResponse({'success': False, 'error': 'No face detected'})
            
            # Largest face
            faces.sort(key=lambda f: (f[2]-f[0])*(f[3]-f[1]), reverse=True)
            embedding = get_face_embedding(frame, faces[0])
            if embedding is None:
                return JsonResponse({'success': False, 'error': 'Embedding failed'})
            
            add_user(name, embedding)
            return JsonResponse({'success': True, 'message': f'Registered {name}'})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    return JsonResponse({'success': False, 'error': 'POST required'})

@csrf_exempt
def login_face(request):
    if request.method == 'POST':
        try:
            # Handle both JSON and form data
            if request.content_type == 'application/json':
                data = json.loads(request.body)
                image_b64 = data.get('image')
            else:
                image_b64 = request.POST.get('image')
            
            frame = base64.b64decode(image_b64.split(',')[1])
            import cv2
            frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
            name = recognize_face(frame)
            
            if name != 'Unknown':
                Attendance.objects.create(name=name)
                return JsonResponse({'status': 'login_success', 'name': name})
            return JsonResponse({'status': 'failed', 'name': name})
        except Exception as e:
            return JsonResponse({'status': 'error', 'name': 'Unknown', 'error': str(e)})
    return JsonResponse({'status': 'error', 'message': 'POST required'})

# Aliases
def recognize(request):
    return ajax_recognize_face(request)

