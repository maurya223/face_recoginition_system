from django.db import models


class User(models.Model):
    name = models.CharField(max_length=100)
    face_embedding = models.BinaryField()

    def __str__(self):
        return self.name


class Login(models.Model):
    name = models.CharField(max_length=100)
    face_embedding = models.BinaryField()

    def __str__(self):
        return self.name


class Attendance(models.Model):
    name = models.CharField(max_length=100)
    date = models.DateField(auto_now_add=True)
    time = models.TimeField(auto_now_add=True)

    def __str__(self):
        return self.name
