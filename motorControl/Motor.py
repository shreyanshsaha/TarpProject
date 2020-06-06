import RPi.GPIO as GPIO
from time import sleep

def debugPrint(*msg):
  return
  # print("[DEBUG]", msg)

class MotorPin():
  def __init__(self, dirPin, pwmPin):
    self.dir = dirPin
    self.pwm = pwmPin
    self.FORWARD=GPIO.HIGH
    self.BACKWARD=GPIO.LOW

    GPIO.setup(self.dir, GPIO.OUT)
    GPIO.setup(self.pwm, GPIO.OUT)
  
  def move(self, direction):
    GPIO.output(self.dir, direction)
    GPIO.output(self.pwm, 1-direction)
  
  def stop(self):
    GPIO.output(self.dir, 0)
    GPIO.output(self.pwm, 0)

  def __del__(self):
    GPIO.output(self.dir, 0)
    GPIO.output(self.pwm, 0)

class MotorController():
  def __init__(self):
    GPIO.setmode(GPIO.BCM)
    self.wheels=[
      MotorPin(2, 3),
      MotorPin(4, 14),
      MotorPin(15, 18),
      MotorPin(17, 27),
      MotorPin(22, 23),
      MotorPin(10, 9),
    ]
    self.FORWARD=1
    self.RIGHT=2
    self.BACKWARD=3
    self.LEFT=4

  def move(self, direction):
    debugPrint("Move", "Direction")
    if direction==self.FORWARD:
      for wheel in self.wheels:
        wheel.move(wheel.FORWARD)
    elif direction==self.RIGHT:
      wheel[0].move(self.FORWARD)
      wheel[1].move(self.FORWARD)
      wheel[2].move(self.FORWARD)
      wheel[3].move(self.BACKWARD)
      wheel[4].move(self.BACKWARD)
      wheel[5].move(self.BACKWARD)
    elif direction==self.BACKWARD:
      for wheel in self.wheels:
        wheel.move(wheel.BACKWARD)
    elif direction==self.LEFT:
      wheel[0].move(self.BACKWARD)
      wheel[1].move(self.BACKWARD)
      wheel[2].move(self.BACKWARD)
      wheel[3].move(self.FORWARD)
      wheel[4].move(self.FORWARD)
      wheel[5].move(self.FORWARD)
  
  def stop(self):
    for wheel in self.wheels:
      wheel.stop()

  