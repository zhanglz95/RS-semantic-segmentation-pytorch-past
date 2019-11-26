import logging

logging.basicConfig(
	level = logging.INFO,
	format = '%(asctime)s - %(levelname)s - %(message)s'
	)

class Logger:
	def __init__(self, name, level):
		self.logger = logging.getLogger()
		handler = logging.FileHandler("./saved/log/{0}.txt".format(name))

		if level == "debug":
			self.logger.setLevel(level=logging.DEBUG)
			handler.setLevel(logging.DEBUG)
		else:
			self.logger.setLevel(level=logging.INFO)
			handler.setLevel(logging.INFO)

		formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
		handler.setFormatter(formatter)
		self.logger.addHandler(handler)

		self.logger.info("Start logging with {0}.".format(level))

	def add_info(self, info):
		self.logger.info(info)