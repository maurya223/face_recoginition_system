from django.test import TestCase, Client
from django.urls import reverse
from .models import Attendance
import json

class HomeTests(TestCase):
    def setUp(self):
        Attendance.objects.create(name='Test User')

    def test_attendance_views(self):
        client = Client()
        
        # Test attendance list
        response = client.get(reverse('attendance'))
        self.assertEqual(response.status_code, 200)
        
        # Test get_attendance
        response = client.get(reverse('get_attendance'))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'attendances', response.content)
        
        # Test manual mark
        response = client.post(reverse('mark_attendance_manual'), {'name': 'New User'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'success', response.content)

