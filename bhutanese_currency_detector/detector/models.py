from django.db import models

class CurrencyImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    result = models.CharField(max_length=100, blank=True)
    
    def __str__(self):
        return f"Currency image {self.id} - {self.result}"