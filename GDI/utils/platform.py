import os
import sys
import tensorflow as tf
try:
	import keras
	keras_use = True 
except ImportError:
	keras_use = False
	pass


class Platform:
	@staticmethod
	def print_line(name, value):
		return "{0:<5s}{1:^20s}:{2:^20s}{3:>5s}\n".format("#", name, value,"#")

	@staticmethod
	def status():
		info = "#"*51+"\n"
		try:
			info += Platform.print_line("ENV Name", os.environ["CONDA_DEFAULT_ENV"])
		except KeyError:
			info += Platform.print_line("ENV Name", "Non conda")
		info += Platform.print_line("Py version", sys.version[:sys.version.find(" ")])
		if keras_use:
			info += Platform.print_line("Keras version", keras.__version__)
		info += Platform.print_line("TF version", tf.__version__)
		info += "#"*51+"\n"
		print(info)


Platform.status()

