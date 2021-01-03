from flask import Flask, request, render_template, session, redirect
from app import app
import pandas as pd
import numpy as np

@app.route('/')
def index():
	return render_template("index.html")

	@app.route('/about')
	def about():
		return render_template("about.html")