import math
import numpy as np
import tools

'''
Assumptions: 
- white - black scale goes from 0 to 255 with black being far away
- width and height pixels are even numbers
- default camera looks in (0,0,1) direction and left side of camera is at (-1,0,0)
- pitch = around y axis, in radians, CCW
- roll = around x axis, in radians, CCW
- yaw = around z axis, in radians, CCW
- depth map image has 0, 0 in the top left corner
- depth map top right positive, bottom left negative


Considerations:
- make sure rotations about i j vectors are in correct direction, currently assuming right hand rule rotation

'''

class Camera:

	# width and height = number of units of length in width and height the camera can see
	# wr and hr = number of radians of width and height the camera can see
	# dark = how far away the camera can "see" using a scale of white to black
	def __init__(self, wr, hr, dark, do_round=True):
		self.wr = wr
		self.hr = hr
		self.darkest = dark
		self.do_round = do_round

	'''
	Given depth image, convert to points in 3D space
	pos = camera x y z in objective 3D space
	projection = 2D numpy matrix of depth values
	p r y = pitch roll yaw of camera's view
	lp, lr, ly = pitch roll raw of left side of camera, needed to determine rotation of frame about direction of view vector
	'''
	def convert(self, pos, projection, view, left_view):
		p, r, y = view
		lp, lr, ly = left_view

		wp = len(projection)
		hp = len(projection[0])
		i_mid = wp/2
		j_mid = hp/2

		#direction of camera's view
		vect = tools.pitch([0,0,1], p, do_round=self.do_round)
		vect = tools.roll(vect, r, do_round=self.do_round)
		vect = tools.yaw(vect, y, do_round=self.do_round)

		#parallel to top and bottom of view frame, runs from right to left
		horz = tools.pitch([-1,0,0], lp, do_round=self.do_round)
		horz = tools.roll(horz, lr, do_round=self.do_round)
		horz = tools.yaw(horz, ly, do_round=self.do_round)

		#parallel to left and right of view frame, runs from bottom to top
		vert = tools.ortho(vect, horz)

		new_pts = set()

		i_temp = np.zeros((8, 8))
		j_temp = np.zeros((8, 8))

		#j = all the rows, i = all the columns
		for j in range(len(projection)):
			for i in range(len(projection[j])):
				#get radians difference from center
				i_rad = -(i + 0.5 - i_mid)/wp * self.wr 
				j_rad = (j + 0.5 - j_mid)/hp * self.hr 

				#get rotation
				m_i = tools.rmatrix_to_vector(tools.invert(tools.make_unit_vector(vert)), j_rad)
				m_j = tools.rmatrix_to_vector(tools.invert(tools.make_unit_vector(horz)), i_rad)

				max_vect = tools.normalize(vect, tools.len_vector(vect))
				max_vect = tools.vector_multiplier(max_vect, self.darkest)

				#get vector from camera to pixel, with vector in objective 3D space
				pix_dist = self.darkest - float(projection[i][j]) / 255 * self.darkest
				pv = [tools.dot_product(vect, m_i[0]), tools.dot_product(vect, m_i[1]), tools.dot_product(vect, m_i[2])]
				pv = [tools.dot_product(pv, m_j[0]), tools.dot_product(pv, m_j[1]), tools.dot_product(pv, m_j[2])]
				pv = tools.make_unit_vector(pv)
				pix_vect = tools.vector_multiplier(pv, pix_dist)

				#get position of this point in 3D space
				pix_pt = tools.sum_vectors(pos, pix_vect)
				if self.do_round:
					new_pts.add(tuple(tools.make_ints(pix_pt)))
				else:
					new_pts.add(tuple(pix_pt))

		return list(new_pts), vert, horz

if __name__ == "__main__":
	pass






















