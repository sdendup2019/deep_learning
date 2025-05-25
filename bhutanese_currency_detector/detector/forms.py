from django import forms
from .models import CurrencyImage

class CurrencyImageForm(forms.ModelForm):
    class Meta:
        model = CurrencyImage
        fields = ['image']
        widgets = {
            'image': forms.FileInput(attrs={'class': 'form-control', 'accept': 'image/*'})
        }