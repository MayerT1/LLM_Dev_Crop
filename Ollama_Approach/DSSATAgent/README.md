"""
TO START THE CELERY WORKER AND BEAT:

1. Start Redis (if using Redis as broker):
   redis-server

2. Start Celery worker:
   celery -A DSSATAgent worker --loglevel=info

3. Start Celery beat (for scheduled tasks, if needed):
   celery -A DSSATAgent beat --loglevel=info

4. For development, you can use:
   celery -A DSSATAgent worker --loglevel=info --pool=solo

5. For production with multiple queues:
   celery -A DSSATAgent worker --loglevel=info -Q chat_processing,celery
"""