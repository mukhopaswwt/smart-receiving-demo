#flaskapp.wsgi
import sys
sys.path.insert(0, '/var/www/html/flaskapp')
#sys.path.insert(0, '/usr/bin/python3.8/')

from flaskapp import app as application