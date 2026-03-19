from django.contrib import admin

# Register your models here.
from .models import User, Login, Attendance

admin.site.register(User)
admin.site.register(Login)
admin.site.register(Attendance)
