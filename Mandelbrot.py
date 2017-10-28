#!/usr/bin/python2

# Author: Elijah Barker
# Date: December 2015

import resource
# import gc
# from guppy import hpy
# import sys
import thread
import time
# import ctypes
# import base64

# PyCUDA imports
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# GUI imports
from Tkinter import *

class DataStorage:
	def __init__(self):
		pass
		self.busy = False
	def save_current(self, image):
		if not self.busy:
			self.busy = True
			with open("Current.ppm","w") as current:
				current.write(image)
			self.busy = False
		else:
			print("Skipped image save.")
#



# Class for computations of mandelbrot set -------------------------------------
class Mandelbrot:
	def __init__(self, zoom_constant, upper_left_r, upper_left_i, pixels_wide, pixels_high, zoom_power_2, max_iterations):
		
		self.precision_32 = 27 - 5
		self.precision_64 = 54 - 3 # 2 if far left pts. are >= -2.0
		self.precision_128 = 108 - 0
		self.kill = False
		
		# maybe make settings somehow?
		self.seconds_per_chunk = .1
		self.upper_chunk_limit = int(1000000 * self.seconds_per_chunk)
				
		self.conf(zoom_constant, upper_left_r, upper_left_i, pixels_wide, pixels_high, zoom_power_2, max_iterations)
		
		self.prepare_gpu()
	#
	def conf(self, zoom_constant, upper_left_r, upper_left_i, pixels_wide, pixels_high, zoom_power_2, max_iterations):
		
		self.zoom_constant = zoom_constant # unit distance between pixels at zoom 0
		self.zoom_power_2 = zoom_power_2
		self.pixels_wide = pixels_wide
		self.pixels_high = pixels_high
		self.max_iterations = max_iterations
		#self.previous_sets = previous_sets
		self.pixelSize = self.zoom_constant/float(2**self.zoom_power_2)
		#print(self.pixelSize)
		self.N = self.pixels_wide*self.pixels_high
		self.M = 32 #32
		
		# starting corner coordinates
		self.lower_left_r = upper_left_r
		self.lower_left_i = upper_left_i - self.pixels_high*self.pixelSize
		
	#
	#
	def prepare_gpu(self):
		
		# allocate arrays
		if self.zoom_power_2 <= self.precision_64:
			if self.zoom_power_2 <= self.precision_32:
				self.zr = np.empty((self.pixels_high,self.pixels_wide),dtype="float32")
				self.zi = np.empty((self.pixels_high,self.pixels_wide),dtype="float32")
				self.cr = np.empty((self.pixels_high,self.pixels_wide),dtype="float32")
				self.ci = np.empty((self.pixels_wide,self.pixels_high),dtype="float32")
			else: # elif self.zoom_power_2 <= self.precision_64:
				self.zr = np.empty((self.pixels_high,self.pixels_wide),dtype="float64")
				self.zi = np.empty((self.pixels_high,self.pixels_wide),dtype="float64")
				self.cr = np.empty((self.pixels_high,self.pixels_wide),dtype="float64")
				self.ci = np.empty((self.pixels_wide,self.pixels_high),dtype="float64")
			
			# set up all coordinates
			#start = time.clock()
			self.cr[0] = np.arange(self.lower_left_r,self.lower_left_r+self.pixels_wide*self.pixelSize,self.pixelSize)
			for k in range(1,self.pixels_high):
				self.cr[k] = self.cr[0]
			#print("Prepare coordinate grid 1:       " + str((time.clock()-start)))
			#start = time.clock()
			self.ci[0] = np.arange(self.lower_left_i+self.pixels_high*self.pixelSize, self.lower_left_i, -self.pixelSize)
			for k in range(1,self.pixels_wide):
				self.ci[k] = self.ci[0]
			self.ci = np.transpose(self.ci).copy() # copy takes 3-8 times as long. maybe use as-is, or calculate on gpu?
			#print("Prepare coordinate grid 2:       " + str((time.clock()-start)))
			
			# set remaining corners
			self.upper_right_r = self.cr[self.pixels_high-1][self.pixels_wide-1]
			self.upper_right_i = self.ci[self.pixels_high-1][self.pixels_wide-1]
			self.lower_right_r = self.cr[0][self.pixels_wide-1]
			self.lower_right_i = self.ci[0][self.pixels_wide-1]
			self.upper_left_r = self.cr[self.pixels_high-1][0]
			self.upper_left_i = self.ci[self.pixels_high-1][0]
			
			# result of mandelbrot computation (Should be zeros, not just empty.)
			self.iterations = np.zeros((self.pixels_high,self.pixels_wide),dtype="int32")
			self.color = np.empty((self.pixels_high, self.pixels_wide, 3),dtype="uint8") # empty
			
			# copy-to/allocate-on gpu
			self.d_zr = cuda.mem_alloc(self.zr.nbytes)
			self.d_zi = cuda.mem_alloc(self.zi.nbytes)
			self.d_cr = cuda.to_device(self.cr)
			self.d_ci = cuda.to_device(self.ci)
			start = time.clock()
			self.d_iterations = cuda.to_device(self.iterations)
			print("Copy single array:               " + str((time.clock()-start)))
			start = time.clock()
			self.d_n = cuda.to_device(np.asarray(np.int32(self.pixels_high*self.pixels_wide)))
			print("Copy single variable:            " + str((time.clock()-start)))
			start = time.clock()
			self.d_color = cuda.mem_alloc(self.color.nbytes)
			print("mem_alloc:                       " + str((time.clock()-start)))
			# max_iterations and color_profile are assumed to be unique for
			# each run of a kernel and will therefore be copied on each run.
			
			
			#elif self.zoom_power_2 <= self.precision_128:
			#	self.cr1 = cuda.mem_alloc()
			#	
			#	self.prepare_128 (
			#		
			#		
			#		block=(self.M,1,1),                       	# block size M
			#		grid=(((self.N+self.M-1)/self.M)*self.M,1)	# (((N+M-1)/M)*M)
			#	)
			
	#
	def calculate(self):
		done = False
		max_iterations = 0
		suggested_iterations = 1024
		
		while not self.kill:
			start = time.clock()
			
			max_iterations = max_iterations + suggested_iterations
			if self.max_iterations <= max_iterations:
				max_iterations = self.max_iterations
				done = True
			
			if self.zoom_power_2 <= self.precision_32:
				self.calc_iterations_32 (
					self.d_n,
					cuda.In(np.asarray(np.int32(max_iterations))),
					self.d_cr,
					self.d_ci,
					self.d_zr,
					self.d_zi,
					self.d_iterations,
					
					block=(self.M,1,1),                       	# block size M
					grid=(((self.N+self.M-1)/self.M)*self.M,1)	# (((N+M-1)/M)*M)
				)
			elif self.zoom_power_2 <= self.precision_64:
				self.calc_iterations_64 (
					self.d_n,
					cuda.In(np.asarray(np.int32(max_iterations))),
					self.d_cr,
					self.d_ci,
					self.d_zr,
					self.d_zi,
					self.d_iterations,
					
					block=(self.M,1,1),                       	# block size M
					grid=(((self.N+self.M-1)/self.M)*self.M,1)	# (((N+M-1)/M)*M)
				)
			
			if done:
				break
			
			elapsed = time.clock()-start
			suggested_iterations = int(suggested_iterations*(self.seconds_per_chunk/elapsed))
			if suggested_iterations > self.upper_chunk_limit:
				suggested_iterations = self.upper_chunk_limit
			if suggested_iterations <= 128:
				suggested_iterations = 128
			print(suggested_iterations)
	#
	def filter_compatible_sets(self, list_of_sets):# so easy
		pass
	def use_compatible_sets(self, list_of_sets):# probably the most complex due to the many possible use cases.
		pass
	
	def gen_image(self, color_profile):
		
		# set header
		self.image = "P6\n#"
		self.image += " "  + str(self.lower_left_r  )
		self.image += " "  + str(self.lower_left_i  )
		self.image += " "  + str(self.zoom_power_2  )
		self.image += " "  + str(self.max_iterations)
		self.image += "\n" + str(self.pixels_wide   )
		self.image += " "  + str(self.pixels_high   )
		self.image += "\n255\n"
		
		# color calculations (cpu option)
		self.color_calculations_gpu(color_profile)
		
		# add colors to data
		self.image=bytearray(self.image)
		self.image += self.color.tostring()
		
		# convert data string to image
		self.img = PhotoImage(data=self.image)
	#
	def color_calculations_cpu(self, color_profile):
		self.iterations = cuda.from_device_like(self.d_iterations, self.iterations)
		color_low = color_profile[0:3]
		color_high = color_profile[3:6]
		color_max = color_profile[6:9]
		max_selector = self.iterations/self.max_iterations
		inverse_selector = 1-max_selector
		for cindex in range(3):
			self.color[:,:,cindex] = ((color_low[cindex] + (color_high[cindex]-color_low[cindex])*self.iterations/self.max_iterations)*(inverse_selector) + color_max[cindex]*max_selector).view("uint8")[:,::4]
	#
	def color_calculations_gpu(self, color_profile):
		self.calc_colors_gpu (
			self.d_n,
			cuda.In(np.array(color_profile).astype(dtype="uint8")),
			cuda.In(np.asarray(np.int32(self.max_iterations))),
			self.d_iterations,
			self.d_color,
			
			block=(self.M,1,1),                       	# block size M
			grid=(((self.N+self.M-1)/self.M)*self.M,1)	# (((N+M-1)/M)*M)
		)
		self.color = cuda.from_device_like(self.d_color, self.color)
	#
	def mem_free(self):
		del self.zr
		del self.zi
		del self.cr
		del self.ci
		del self.upper_right_r
		del self.upper_right_i
		del self.lower_right_r
		del self.lower_right_i
		del self.upper_left_r
		del self.upper_left_i
		del self.iterations
		del self.color
		del self.d_zr
		del self.d_zi
		del self.d_cr
		del self.d_ci
		del start
		del self.d_iterations
		del self.d_n
		del self.d_color
	#
#









# Logic Layer Class ------------------------------------------------------------
class Controller:
	def __init__(self, canvas):
		self.canvas = canvas
		self.default_settings()
		self.compile_kernels()
		self.data_storage = DataStorage()
		# initialize list of Mandelbrot objects
		self.mandelbrot_sets = []
	#
	
	def default_settings(self):
		# performance settings
		self.max_cache = 1 # number of mandelbrot sets to cache
		self.apply_changes_on_canvas_changed = False
		self.update_image_async = False
		self.save_image = True
		
		# for testing only?
		self.zoom_constant = 1 # unit distance between pixels at zoom 0
		self.pan_speed = 1
		self.zoom_speed = 1
		self.zoom_reset()
		self.max_iterations_reset()
		self.position_reset()
		self.controls_hidden = False
		self.drag = False
		self.fullscreen = False
		self.window_size = "1280x720" #"400x400"
		self.color_profile = [160,38,48,255,255,255,96,16,90] # [160,38,48,255,255,255,96,16,90]  [0,0,100,255,255,255,0,0,0]  [200,0,80,255,255,255,70,0,100]  [0,0,0],[0,0,0],[0,0,0]  [255,255,255],[255,255,255],[255,255,255]
	#
	
	def compile_kernels(self):
		# prepare GPU methods
		#start = time.clock()
		code_module = SourceModule("""
			__global__ void calc_iterations_32 (int *n, int *max_iterations, float *cRa, float *cIa, float *zRa, float *zIa, int *iterations_data)
			{
				//*
				int index = blockIdx.x*blockDim.x + threadIdx.x;
				if (index < *n)
				{
					
					const float cR = cRa[index];
					const float cI = cIa[index];
					int iterator = iterations_data[index];
					float zR;
					float zI;
					if(iterator == 0)
					{
						zR = 0;
						zI = 0;
					}
					else
					{
						zR = zRa[index];
						zI = zIa[index];
					}
					float zR2; //for optimization
					float zI2; //for optimization
					while (iterator < *max_iterations)
					{
						zR2 = zR*zR;
						zI2 = zI*zI;
						if (zR2 + zI2 >= 4) break;
						zI = cI + 2*zR*zI;
						zR = cR + zR2 - zI2;
						iterator++;
					}
					iterations_data[index] = iterator;
					zRa[index] = zR;
					zIa[index] = zI;
				}
				//*/
			}
			__global__ void calc_iterations_64 (int *n, int *max_iterations, double *cRa, double *cIa, double *zRa, double *zIa, int *iterations_data)
			{
				//*
				int index = blockIdx.x*blockDim.x + threadIdx.x;
				if (index < *n)
				{
					
					const double cR = cRa[index];
					const double cI = cIa[index];
					int iterator = iterations_data[index];
					double zR;
					double zI;
					if(iterator == 0)
					{
						zR = 0;
						zI = 0;
					}
					else
					{
						zR = zRa[index];
						zI = zIa[index];
					}
					double zR2; //for optimization
					double zI2; //for optimization
					while (iterator < *max_iterations)
					{
						zR2 = zR*zR;
						zI2 = zI*zI;
						if (zR2 + zI2 >= 4) break;
						zI = cI + 2*zR*zI;
						zR = cR + zR2 - zI2;
						iterator++;
					}
					iterations_data[index] = iterator;
					zRa[index] = zR;
					zIa[index] = zI;
				}
				//*/
			}
			/*
			__global__ void prepare_128 (int *n, int *zoom, int *nDoubles, double *upper_left_r, double *upper_left_i, double *cRa, double cIa)
			{
				int index = blockIdx.x*blockDim.x + threadIdx.x;
				if (index < *n)
				{
					
				}
			}//*/
			__global__ void calc_colors_gpu (int *n, unsigned char *color_profile, int *max_iterations, int *iterations_data, unsigned char *color_data)
			{
				int index = blockIdx.x*blockDim.x + threadIdx.x;
				if (index < *n)
				{
					const int max_selector = iterations_data[index]/(*max_iterations);
					const int inverse_selector = 1-max_selector;
					for(int i=0;i<3;i++)
					{
						color_data[index*3+i] = (unsigned char)((color_profile[i] + ((color_profile[3+i]-color_profile[i])*(iterations_data[index]))/(*max_iterations))*(inverse_selector) + color_profile[6+i]*(max_selector));
					}
				}
			}
		""")
		self.calc_iterations_32 = code_module.get_function("calc_iterations_32")
		self.calc_iterations_64 = code_module.get_function("calc_iterations_64")
		#self.prepare_128 = code_module.get_function("prepare_128")
		self.calc_colors_gpu = code_module.get_function("calc_colors_gpu")
		#print("Compile CUDA code:               " + str((time.clock()-start)))
	#
	def canvas_changed(self, width, height):
		self.width = width
		self.height = height
		self.reset_xy()
		if self.apply_changes_on_canvas_changed:
			self.apply_changes()
	#
	
	def compute_mandelbrot(self):
		#del(self.mandelbrot)
		start = time.clock()
		# if hasattr(self, 'mandelbrot'):
		# 	self.mandelbrot.mem_free()
		# 	del self.mandelbrot
		# gc.collect()
		print ('Memory usage before: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
		self.mandelbrot = Mandelbrot(self.zoom_constant, self.posR, self.posI, self.width, self.height, self.zoom, self.max_iterations)
		print ('Memory usage after: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
		self.mandelbrot.calc_iterations_32 = self.calc_iterations_32
		self.mandelbrot.calc_iterations_64 = self.calc_iterations_64
		self.mandelbrot.calc_colors_gpu    = self.calc_colors_gpu
		
		print("Init prepare:            " + str((time.clock()-start)))
		self.mandelbrot.canvas = self.canvas # for testing
		start = time.clock()
		self.mandelbrot.calculate()
		print("Iterations calculations: " + str((time.clock()-start)))
		start = time.clock()
		self.mandelbrot.gen_image(self.color_profile) # [160,38,48],[255,255,255],[96,16,90]  [0,0,100],[255,255,255],[0,0,0]  [200,0,80],[255,255,255],[70,0,100]  [0,0,0],[0,0,0],[0,0,0]  [255,255,255],[255,255,255],[255,255,255]
		print("Generate Image:          " + str((time.clock()-start)))
		#start = time.clock()
		self.save_mandelbrot()
		#print("Save Image to file:      " + str((time.clock()-start)))
		
		#start = time.clock()
		# if len(self.mandelbrot_sets) > self.max_cache-1:# not 'smart' storage, make own method
		# 	self.mandelbrot_sets.pop()
		# self.mandelbrot_sets.append(self.mandelbrot) # memory bomb! (if without safeguard above)
		#print("Cache Mandelbrot set:    " + str((time.clock()-start)))
		
		
	#
	def display_mandelbrot(self):
		#self.canvas.delete("all")
		if self.update_image_async:
			thread.start_new_thread(self.canvas.create_image, (self.width/2,self.height/2), {"image" : self.mandelbrot.img})
		else:
			start = time.clock()
			self.canvas.create_image((self.width/2,self.height/2),image=self.mandelbrot.img)
			print("Display image on canvas: " + str((time.clock()-start)))
	#
	def kill(self):
		self.mandelbrot.kill = True
	#
	def save_mandelbrot(self):
		if self.save_image:
			thread.start_new_thread(self.data_storage.save_current, (self.mandelbrot.image,))
	#
	def increase_iterations(self):
		self.max_iterations *= 2
		#print (self.max_iterations)
	#
	def decrease_iterations(self):
		self.max_iterations /= 2
		#print (self.max_iterations)
	#
	def max_iterations_reset(self):
		self.max_iterations = 100
	#
	def increase_zoom_speed(self):
		self.zoom_speed += 1
	#
	def decrease_zoom_speed(self):
		self.zoom_speed -= 1
	#
	def zoom_reset(self):
		self.zoom = 8
		self.update_pixelSize()
	#
	def increase_pan_speed(self):
		self.pan_speed += 1
	#
	def decrease_pan_speed(self):
		self.pan_speed -= 1
	#
	def position_reset(self):
		self.posR = -3.47265625 # upper left 
		self.posI = 1.15625 # upper left
	#
	def zoom_in(self):
		#print(str(self.mouse_x) + " " + str(self.mouse_y))
		self.posR += (self.mouse_x) * self.pixelSize # center of new screen
		self.posI -= (self.mouse_y) * self.pixelSize # center of new screen
		
		self.zoom += self.zoom_speed
		print("Zoom: "+str(self.zoom))
		self.update_pixelSize()
		
		self.posR -= self.width  * self.pixelSize / 2
		self.posI += self.height * self.pixelSize / 2
		
		# strip precision for lower zooms (guarantees number to index mappings for optimizations)
		self.posR = int(self.posR * 2**self.zoom)/float(2**self.zoom)
		self.posI = int(self.posI * 2**self.zoom)/float(2**self.zoom)
	#
	def zoom_out(self):
		#print(str(self.mouse_x) + " " + str(self.mouse_y))
		self.posR += (self.mouse_x) * self.pixelSize # center of new screen
		self.posI -= (self.mouse_y) * self.pixelSize # center of new screen
		
		self.zoom -= self.zoom_speed
		print("Zoom: "+str(self.zoom))
		self.update_pixelSize()
		
		self.posR -= self.width  * self.pixelSize / 2
		self.posI += self.height * self.pixelSize / 2
		
		# strip precision for lower zooms (guarantees number to index mappings for optimizations)
		self.posR = int(self.posR * 2**self.zoom)/float(2**self.zoom)
		self.posI = int(self.posI * 2**self.zoom)/float(2**self.zoom)
	#
	def pan(self, x, y):
			self.posR -= (x - self.mouse_x) * self.pixelSize * self.pan_speed
			self.posI += (y - self.mouse_y) * self.pixelSize * self.pan_speed
	#
	def update_pixelSize(self):
		self.pixelSize = self.zoom_constant/float(2**self.zoom)
	#
	def apply_changes(self):
		start = time.clock()
		#print (self.max_iterations)
		self.compute_mandelbrot()
		self.display_mandelbrot()
		print("Apply Changes:         = " + str((time.clock()-start)))
		print("Apply Changes:         = " + str(int(100/(time.clock()-start))/100.0) + " fps")
		print(str(self.mandelbrot.upper_left_i) + " " + str(self.mandelbrot.upper_left_r))
	#
	def reset_xy(self):
		self.mouse_x = self.width/2
		self.mouse_y = self.height/2
	#
	def left_mouse_down(self, x, y):
		self.mouse_x = x
		self.mouse_y = y
		self.drag = False
		#print(str(self.mouse_x) + " " + str(self.mouse_y))
	#
	def left_mouse_drag(self, x, y):
		#print(str(x) + " " + str(y))
		self.pan(x, y)
		self.mouse_x = x
		self.mouse_y = y
		self.drag = True
		
		self.apply_changes()
	#
	def left_mouse_up(self, x, y):
		if not self.drag:
			self.zoom_in()
			self.apply_changes()
		self.reset_xy()
	#
	def right_mouse_up(self, x, y):
		if not self.drag:
			self.zoom_out()
			self.apply_changes()
		self.reset_xy()
	#
#











# GUI (View) Class -------------------------------------------------------------
class MandelbrotApp:
	
	def __init__ (self, parent):
		
		self.window = parent
		self.canvas = Canvas(self.window, highlightthickness=0)
		self.logic = Controller(self.canvas)
		
		# window
		self.set_window_size(self.logic.window_size, self.logic.fullscreen)
		self.window.wm_title("Mandelbrot Fractal")
		
		# control frame
		self.control_frame = Frame(self.window)
		
		# canvas (no frame)
		self.canvas.configure(background="#%02x%02x%02x" % (self.logic.color_profile[0], self.logic.color_profile[1], self.logic.color_profile[2]))
		self.canvas.pack(side=BOTTOM, expand=YES, fill=BOTH)
		self.canvas.bind("<Configure>", self.canvas_changed)
		self.canvas.bind("<Button-1>", self.left_button_down)
		self.canvas.bind("<B1-Motion>", self.left_button_drag)
		self.canvas.bind("<ButtonRelease-1>", self.left_button_up)
		self.canvas.bind("<Button-3>", self.left_button_down)
		self.canvas.bind("<B3-Motion>", self.left_button_drag)
		self.canvas.bind("<ButtonRelease-3>", self.right_button_up)
		self.canvas.bind("<Key>", self.key_pressed)
		self.canvas.width = 0
		self.canvas.height = 0
		
		self.init_buttons()
		
		self.show_controls()
		
		# print ("exited constructor")
	#
	def init_buttons(self):
		nButtons = 16
		mColumns = 4
		
		self.buttons=range(nButtons)
		for i in range(nButtons):
			self.buttons[i] = Button(self.control_frame)
			self.buttons[i].grid(row=i/mColumns, column=i%mColumns, sticky=E+W)
		#
		for i in range(mColumns):
			self.control_frame.columnconfigure(i, weight=1)
		#
		self.update_buttons()
	#
	def update_buttons(self):
		self.buttons[ 2].configure(text="Fullscreen  ( f )",command=self.toggle_fullscreen)
		self.buttons[ 3].configure(text="Exit  ( Esc )",command=self.window.destroy)
		self.buttons[12].configure(text="Apply Changes  ( Enter, Space )",command=self.apply_changes)
		self.buttons[ 5].configure(text="Iterations " + str(self.logic.max_iterations) + " * 2  ( i )",command=self.increase_iterations)
		self.buttons[ 9].configure(text="Iterations " + str(self.logic.max_iterations) + " / 2  ( k )",command=self.decrease_iterations)
		self.buttons[13].configure(text="Reset Max Iterations  ( r )",command=self.reset_iterations)
		self.buttons[ 6].configure(text="Zoom Speed " + str(self.logic.zoom_speed) + " + 1  ( o )",command=self.increase_zoom_speed)
		self.buttons[10].configure(text="Zoom Speed " + str(self.logic.zoom_speed) + " - 1  ( l )",command=self.decrease_zoom_speed)
		self.buttons[14].configure(text="Zoom " + str(self.logic.zoom) + " Reset  ( r )",command=self.reset_zoom)
		self.buttons[ 7].configure(text="Pan Speed " + str(self.logic.pan_speed) + " + 1  ( p )",command=self.increase_pan_speed)
		self.buttons[11].configure(text="Pan Speed " + str(self.logic.pan_speed) + " - 1  ( ; )",command=self.decrease_pan_speed)
		self.buttons[ 8].configure(text="Abort Calculations")#,command=self.)
		self.buttons[ 1].configure(text="Hide Controls  ( c )",command=self.toggle_controls)
		self.buttons[ 4].configure(text="Settings (not yet functional)")#,command=self.)
		self.buttons[ 0].configure(text="Load Settings from File (not yet functional)")#,command=self.)
		self.buttons[15].configure(text="Position Reset  ( r )",command=self.reset_position)
		
		if self.logic.controls_hidden:
			self.hide_controls()
		else:
			self.show_controls()
		
		self.canvas.focus_set()
	#
	def key_pressed(self, event):
		
		if event.char == "r":
			self.reset_zoom()
			self.reset_iterations()
			self.reset_position()
		elif event.char == "f":
			self.toggle_fullscreen()
		elif event.char == "c":
			self.toggle_controls()
		elif event.char == "i":
			self.increase_iterations()
		elif event.char == "k":
			self.decrease_iterations()
		elif event.char == "o":
			self.increase_zoom_speed()
		elif event.char == "l":
			self.decrease_zoom_speed()
		elif event.char == "p":
			self.increase_pan_speed()
		elif event.char == ";":
			self.decrease_pan_speed()
		elif event.char == " ":
			self.apply_changes()
		elif event.char == "":
			print ("Empty key press")
		elif ord(event.char) == 13:
			self.apply_changes()
		elif ord(event.char) == 27:
			self.window.destroy()
		else:
			print("key_pressed: " + event.char + " " + str(ord(event.char)))
	#
	def toggle_controls(self):
		self.logic.controls_hidden = not self.logic.controls_hidden
		self.update_buttons()
	#
	def show_controls(self):
		self.control_frame.pack(side=TOP, fill=X)
		self.canvas.pack(side=BOTTOM, expand=YES, fill=BOTH)
	#
	def hide_controls(self):
		self.control_frame.pack_forget()
	#
	def set_window_size(self, window_size, fullscreen):
		if fullscreen:
			self.logic.window_size = self.window.winfo_geometry()
			self.window.geometry("{0}x{1}+0+0".format(self.window.winfo_screenwidth(), self.window.winfo_screenheight()))
		else:
			self.window.geometry(self.logic.window_size)
		#
		self.window.attributes('-fullscreen', self.logic.fullscreen)
		
		# self.window.overrideredirect(self.fullscreen) # does not work right
		#self.window.withdraw() # not needed
		#self.window.deiconify() # not needed
	#
	def toggle_fullscreen(self):
		self.logic.fullscreen = not self.logic.fullscreen
		self.set_window_size(self.logic.window_size, self.logic.fullscreen)
	#
	def increase_iterations(self):
		self.logic.increase_iterations()
		self.update_buttons()
	#
	def decrease_iterations(self):
		self.logic.decrease_iterations()
		self.update_buttons()
	#
	def reset_iterations(self):
		self.logic.max_iterations_reset()
		self.update_buttons()
	#
	def increase_zoom_speed(self):
		self.logic.increase_zoom_speed()
		self.update_buttons()
	#
	def decrease_zoom_speed(self):
		self.logic.decrease_zoom_speed()
		self.update_buttons()
	#
	def reset_zoom(self):
		self.logic.zoom_reset()
		self.update_buttons()
	#
	def increase_pan_speed(self):
		self.logic.increase_pan_speed()
		self.update_buttons()
	#
	def decrease_pan_speed(self):
		self.logic.decrease_pan_speed()
		self.update_buttons()
	#
	def reset_position(self):
		self.logic.position_reset()
	#
	def apply_changes(self):
		self.logic.apply_changes()
		# print sys.getsizeof(logic)
		# h = hpy()
		# print h.heap()
		print (vars(self))
	#
	def canvas_changed(self, event):
		#print ("canvas_changed()")
		self.logic.canvas_changed(event.width, event.height)
	#
	def left_button_down(self, event):
		self.logic.left_mouse_down(event.x, event.y)
		self.update_buttons()
		#print(str(event.x) + " " + str(event.y))
	#
	def left_button_drag(self, event):
		self.logic.left_mouse_drag(event.x, event.y)
		self.update_buttons()
		#print(str(event.x) + " " + str(event.y))
	#
	def left_button_up(self, event):
		self.logic.left_mouse_up(event.x, event.y)
		self.update_buttons()
		#print(str(event.x) + " " + str(event.y))
	#
	def right_button_up(self, event):
		self.logic.right_mouse_up(event.x, event.y)
		self.update_buttons()
	#
#

# Run program
def main():
	root = Tk()
	mandelbrotApp = MandelbrotApp(root)
	root.mainloop()

if __name__ == "__main__":
	main()


# address chunk of array:  myArray[0:3:1,0:3:1]

#Cool thing
#Iterations: 2048
#Zoom: 48
#0.640431646287 -0.559407104593

#a02630 # [160,38,48],[255,255,255],[96,16,90]
#255,255,255
#63105a # 96,16,90
