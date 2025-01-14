from django.db import models

class Observation(models.Model):
    date = models.DateField()
    precipitation = models.FloatField()
    temp_max = models.FloatField()
    temp_min = models.FloatField()
    wind = models.FloatField()
    humidity = models.FloatField()
    pressure = models.FloatField()
    solar_radiation = models.FloatField()
    visibility = models.FloatField()
    weather_id = models.IntegerField()
    cloudiness_id = models.IntegerField()
    season_id = models.IntegerField()

    def __str__(self):
        return f"Observation on {self.date}"
