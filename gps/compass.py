
# ! ===========================================
# ! DO NOT CHANGE THIS FILE, IT WORKS PERFECTLY
# ! CHANGE AUTONOMOUS.PY IF NEEDED
# ! ===========================================
from serial import Serial
import time
class Compass(Serial):
	def __init__(self, device):
		self.devicePath=device
		self.connectCompass()
		self.value=0
		self.i=0
	
	def connectCompass(self):
		try:
			self.compassSerial = Serial(self.devicePath)
		except Exception as e:
			raise e

	def getCompassAngle(self):
		try:
				self.value = int(str(self.compassSerial.readline().rstrip())[2:-1])
				if 180<self.value and self.value<=360:
					self.value = self.value - 360
				time.sleep(100)
				return self.value
		except TypeError as e:
			raise e
		except ValueError:
			pass
	
	def resetCompass(self):
		self.compassSerial.close()
		self.compassSerial.flush()
		self.compassSerial.open()
		
		if not self.compassSerial.is_open:
			raise "Cannot reopen compass serial"
# ! ======================
# ! DO NOT CHANGE THE FILE
# ! ======================


# com = Compass("/dev/ttyACM0")
# while True:
# 		print(com.getCompassAngle())
