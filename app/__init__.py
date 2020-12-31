from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config import app_config
from flas_login import LoginManager

db = SQLAlchemy()
login_manager = LoginManager()

def create_app(config_name) : 
	app = Flask(__name__, instance_relative_config=True)
	app.config.from_object(app_config[config_name])
	app.config.from_pyfile('config.py')
	#db_init_app(app)
	@app.route('/')

def hello_world():
	return 'Hello, World!'

def create_app(config_name):
	login_manageer.init_app(app)
	login_manager.login_message = "You must be logged in to acces this page."
	login_manager.login_view = "auth.login"
	return app