"""
	Annotation tool

		Tool used to annotate labels for LP detection. Main keys:
			C: creates a new shape
			A: creates a new vertex over mouse position
			D: delete last vertex
			S: change position of closest vertex (to mouse pos.)
			X: switch to closest shape
			N or enter: next picture
			P: previous picture
			For more commands, take a look at the main part...

		Usage:

			python annotation-tool.py max_width max_height file1 ... fileN
		
		Obs. This tool is not fully tested and crashes under unexpected situations.
		If you find any and correct, feel free to send a pull request =)

"""

import cv2
import sys
import numpy as np
import os

from math import cos, sin
from os.path import isfile, splitext, basename, isdir
from os import makedirs

from src.utils import image_files_from_folder,getWH
from src.label import Label, Shape, readShapes, writeShapes
from src.projection_utils import perspective_transform, find_T_matrix, getRectPts


class ShapeDisplay(Shape):

	def appendSide(self,pt):
		if (self.max_sides == 0) or (self.pts.shape[1] < self.max_sides):
			self.pts = np.append(self.pts,pt,axis=1)

	def removeLast(self):
		self.pts = self.pts[...,:-1]

	def changeClosest(self,pt):
		idx = np.argmin(self.distanceTo(pt))
		self.pts[...,idx] = pt[...,0]

	def distanceTo(self,pt):
		return np.sqrt(np.power((self.pts - pt),2).sum(0))

	def shiftPts(self):
		if self.pts.shape[1] > 1:
			idx = list(range(1,self.pts.shape[1])) + [0]
			self.pts = self.pts[...,idx]

	def getSquare(self):
		tl,br = self.pts.min(1),self.pts.max(1)
		return Label(-1,tl,br)

	def draw(self,drawLineFunc,drawCircleFunc,drawTextFunc,color=(255,255,255),txtcolor=(255,255,255)):
		ss = self.pts.shape[1]
		if ss:
			text = self.text if len(self.text) else 'NO_LABEL'
			drawTextFunc(text,self.pts.min(1),color=txtcolor)
		if ss > 1:
			for i in range(ss):
				drawLineFunc(self.pts[:,i],self.pts[:,(i+1)%ss],color=color)
				drawCircleFunc(self.pts[:,0],color=(255-color[0],0,255-color[2]))
				drawCircleFunc(self.pts[:,-1],color=(255-color[0],255-color[1],255-color[2]))


def rotation_transform(wh,angles=np.array([0.,0.,0.]),zcop=1000., dpp=1000.):
	rads = np.deg2rad(angles)
	a = rads[0]; Rx = np.matrix([[1,0,0],[0,cos(a),sin(a)],[0,-sin(a),cos(a)]])
	a = rads[1]; Ry = np.matrix([[cos(a),0,-sin(a)],[0,1,0],[sin(a),0,cos(a)]])
	a = rads[2]; Rz = np.matrix([[cos(a),sin(a),0],[-sin(a),cos(a),0],[0,0,1]])
	return Rx*Ry*Rz

class Display():

	def __init__(self,I,width,height,wname='Display'):
		self.Iorig = I.copy()
		self.width = width
		self.height = height
		self.wname = wname
		self.Idisplay = self.Iorig.copy()
		self.IdisplayCopy = self.Idisplay.copy()
		self.reset_view()
		self._setPerspective()
		cv2.namedWindow(self.wname)
		cv2.moveWindow(self.wname,0,0)
		cv2.setMouseCallback(self.wname,self.mouse_callback)

	def reset_view(self):
		self.cx,self.cy = .5,.5
		wh = np.array([self.width,self.height],dtype=float)
		self.zoom_factor = (wh/getWH(self.Iorig.shape)).min()
		self.mouse_center = np.array([.5,.5])
		self.angles = np.array([0.,0.,0.])
		self._setPerspective()

	def updatePerspectiveMatrix(self):
		zf = self.zoom_factor
		w,h = getWH(self.Iorig.shape)
		self.dx = self.cx*w*zf - self.width/2.
		self.dy = self.cy*h*zf - self.height/2.
		R = np.eye(3)
		R = np.matmul(R,np.matrix([[zf,0,-self.dx],[0,zf,-self.dy],[0,0,1]],dtype=float))
		R = np.matmul(R,perspective_transform((w,h),angles=self.angles))
		self.R = R
		self.Rinv = np.linalg.inv(R)

	def show(self):
		cv2.imshow(self.wname,self.Idisplay)

	def setPerspectiveAngle(self,addx=0.,addy=0.,addz=0.):
		self.angles += np.array([addx,addy,addz])
		self._setPerspective()

	def _setPerspective(self,update=True):
		if update: self.updatePerspectiveMatrix()
		self.IdisplayCopy = cv2.warpPerspective(self.Iorig,self.R,(self.width,self.height),borderValue=.0,flags=cv2.INTER_LINEAR)

	def resetDisplay(self):
		self.Idisplay = self.IdisplayCopy.copy()

	def getMouseCenterRelative(self):
		return self.mouse_center.copy().reshape((2,1))

	def waitKey(self,time=50):
		return cv2.waitKey(50) & 0x0000000FF

	def __pt2xy(self,pt):
		pt = np.squeeze(np.array(np.matmul(self.R,np.append(pt*getWH(self.Iorig.shape),1.))))
		pt = pt[:2]/pt[2]
		return tuple(pt.astype(int).tolist())

	def __pts2xys(self,pts):
		N = pts.shape[1]
		pts = pts*getWH(self.Iorig.shape).reshape((2,1))
		pts = np.concatenate((pts,np.ones((1,N))))
		pts = np.squeeze(np.array(np.matmul(self.R,pts)))
		pts = pts[:2]/pts[2]
		return pts

	def drawLine(self,pt1,pt2,color=(255,255,255),thickness=3):
		pt1 = self.__pt2xy(pt1); pt2 = self.__pt2xy(pt2)
		cv2.line(self.Idisplay,pt1,pt2,color=color,thickness=thickness)

	def drawCircle(self,center,color=(255,255,255),radius=7):
		center = self.__pt2xy(center)
		cv2.circle(self.Idisplay,center,radius,color,thickness=-1)

	def drawText(self,text,bottom_left_pt,color=(255,255,255),bgcolor=(0,0,0),font_size=1):
		bl_corner = self.__pt2xy(bottom_left_pt)
		font = cv2.FONT_HERSHEY_SIMPLEX
		wh_text,v = cv2.getTextSize(text,font,font_size,3)
		tl_corner = (bl_corner[0],bl_corner[1]-wh_text[1])
		br_corner = (bl_corner[0]+wh_text[0],bl_corner[1])
		cv2.rectangle(self.Idisplay,tl_corner,br_corner,bgcolor,-1)
		cv2.putText(self.Idisplay,text,bl_corner,font,font_size,color,3)

	def zoom(self,ff):
		self.zoom_factor *= ff
		self.cx,self.cy = self.mouse_center.tolist()
		self._setPerspective()

	def rectifyToPts(self,pts):
		if pts.shape[1] != 4: return
		ptsh = pts*getWH(self.Iorig.shape).reshape((2,1))
		ptsh = np.concatenate((ptsh,np.ones((1,4))))
		to_pts = self.__pts2xys(pts)
		wi,hi = (to_pts.min(1)[:2]).tolist()
		wf,hf = (to_pts.max(1)[:2]).tolist()
		to_pts = np.matrix([[wi,wf,wf,wi],[hi,hi,hf,hf],[1,1,1,1]])
		self.R = find_T_matrix(ptsh,to_pts)
		self.Rinv = np.linalg.inv(self.R)
		self._setPerspective(update=False)

	def mouse_callback(self,event,x,y,flags,param):
		mc = np.array([x,y],dtype=float)
		mc = np.matmul(self.Rinv,np.append(mc,1.))
		mc = np.squeeze(np.array(mc))
		self.mouse_center = (mc[:2]/mc[2])/getWH(self.Iorig.shape)

def selectClosest(shapes,pt):
	if len(shapes):
		mindist,selected = shapes[0].distanceTo(pt).min(),0
		for i,shape in enumerate(shapes[1:]):
			shpdist = shape.distanceTo(pt).min()
			if mindist > shpdist:
				selected = i+1; mindist = shpdist
		return selected
	return -1

def displayAllShapes(disp,shapes,selected,typing_mode):
	for i,shape in enumerate(shapes):
		color = (255,255,255) if i != selected else (0,0,255)
		txtcolor = (0,0,255) if (i == selected and typing_mode) else (255,255,255)
		shape.draw(disp.drawLine,disp.drawCircle,disp.drawText,color=color,txtcolor=txtcolor)

if __name__ == '__main__':
	if len(sys.argv) < 4:
		print(__doc__); sys.exit()
	maxW = int(sys.argv[1]); maxH = int(sys.argv[2]); img_files = sys.argv[3:]
	ENTER = 10; ESC = 27; BACKSPACE = 8
	key_exit = ESC
	key_next = [ord('n'),ENTER]
	key_previous = ord('p')
	action_keys = [key_exit,key_previous] + key_next
	curr_image = 0
	while curr_image < len(img_files):
		img_file = img_files[curr_image]
		lab_file = splitext(img_file)[0] + '.txt'
		if isfile(lab_file):
			shapes = readShapes(lab_file,obj_type=ShapeDisplay)
			selected = len(shapes)-1
		else:
			shapes,selected = [ShapeDisplay()],0
		disp = Display(cv2.imread(img_file),maxW,maxH)
		disp.show()
		key = disp.waitKey()
		typing_mode = False
		while key not in action_keys:
			disp.resetDisplay()
			displayAllShapes(disp,shapes,selected,typing_mode)
			disp.show()
			key = disp.waitKey(10)
			if typing_mode:
				if key == ord(' '):
					typing_mode = False
				elif key != 255:
					if key >= 176: key = key - 176 + 48
					if key == BACKSPACE:
						shapes[selected].text = shapes[selected].text[:-1]
					else:
						shapes[selected].text += str(chr(key)).upper()
				key = 255
				continue
			if key == ord('q'): disp.zoom(1.5)
			if key == ord('w'): disp.zoom(.5)
			if key == ord('a'):
				print('Append vertex')
				shapes[selected].appendSide(disp.getMouseCenterRelative())
			if key == ord('d'):
				print('Remove last vertex')
				shapes[selected].removeLast()
			if key == ord('s'):
				print('Change closest vertex')
				shapes[selected].changeClosest(disp.getMouseCenterRelative())
			if key == ord('c'):
				print('Create new shape')
				shapes.append(ShapeDisplay())
				selected = len(shapes)-1
			if key == ord('x'):
				print('Select closest')
				pt = disp.getMouseCenterRelative()
				selected = selectClosest(shapes,pt)
			if key == ord('r'):
				print('Delete selected shape')
				del shapes[selected]
				pt = disp.getMouseCenterRelative()
				selected = selectClosest(shapes,pt)
			if key == ord(' '):
				typing_mode = True
			if key == ord('g'):
				shapes[selected].shiftPts()
			if key == ord('l'):
				disp.rectifyToPts(shapes[selected].pts)
			if key == 81: disp.setPerspectiveAngle(addy=10.)
			if key == 83: disp.setPerspectiveAngle(addy=-10.)
			if key == 82: disp.setPerspectiveAngle(addx=10.)
			if key == 84: disp.setPerspectiveAngle(addx=-10.)
			if key == 46: disp.setPerspectiveAngle(addz=10.)
			if key == 44: disp.setPerspectiveAngle(addz=-10.)
			if key == 80: disp.reset_view()
		if key == key_exit: sys.exit()
		if key in ([key_previous] + key_next):
			writeShapes(lab_file,shapes)
			curr_image += 1 if key in key_next else -1
			curr_image = max(curr_image,0)
