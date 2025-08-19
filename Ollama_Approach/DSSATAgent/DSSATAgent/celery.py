import os
from celery import Celery

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'DSSATAgent.settings')

app = Celery('DSSATAgent')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django app configs.
app.autodiscover_tasks()

# Celery beat schedule (if you need periodic tasks)
app.conf.beat_schedule = {
    # Example: Clean up old failed messages every hour
    # 'cleanup-failed-messages': {
    #     'task': 'webapp.tasks.cleanup_failed_messages',
    #     'schedule': 3600.0,  # 1 hour
    # },
}

app.conf.timezone = 'UTC'


@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')


