/*
 * based on https://gist.github.com/je-so/903479 - copyright (c) 2011 Joerg Seebohn
 *
 * modified/updated (c) 2021 by Florian 'floe' Echtler <floe@butterbrot.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * compile with: g++ -Wall -o tv transparent_viewer.c -lX11 -lGL $(pkg-config --cflags --libs opencv4)
*/

#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/keysym.h>

#include <GL/gl.h>
#include <GL/glx.h>

#include <stdio.h>
#include <stdlib.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio_c.h>

#include <vector>

// global vars (yuck)
Display* display;
Window win;
cv::VideoCapture cap;

int width = 640;
int height = 480;

int x_wait_for_events( int msec ) {
	fd_set fdset;
	struct timeval wait;

	int socket = ConnectionNumber( display );
	FD_ZERO( &fdset );
	FD_SET( socket, &fdset );

	wait.tv_sec = msec / 1000;
	wait.tv_usec = (msec % 1000) * 1000;
	int err = select( socket+1, &fdset, NULL, NULL, &wait );

	return (( err < 0 ) && ( errno != EINTR )) ? -1 : 0;
}

void gl_init_texture() {
	GLuint t = 0;

	glEnable(GL_TEXTURE_2D);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	glGenTextures( 1, &t );
	glBindTexture(GL_TEXTURE_2D, t);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
}

void gl_update_texture() {
	cv::Mat pic,alpha;
	cap.read(pic);
	cv::cvtColor(pic,alpha,cv::COLOR_YUV2RGBA_YUYV);

	uint32_t* data = (uint32_t*)(alpha.data);
	for (size_t i = 0; i < alpha.total(); i++) {
		// chroma key on 100% green
		if ((data[i] & 0x0000FF00) == 0x0000FF00)
			data[i] = 0;
	}

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, alpha.cols, alpha.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, alpha.data );
}

void gl_idle() {
	glClearColor( 1.0, 1.0, 1.0, 1.0 );
	glClear(GL_COLOR_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION); glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);  glLoadIdentity();

	glBegin(GL_QUADS);
		glTexCoord2f(0.0, 0.0);
		glVertex2i(-1, 1);
		glTexCoord2f(1.0, 0.0);
		glVertex2i(1, 1);
		glTexCoord2f(1.0, 1.0);
		glVertex2i(1, -1);
		glTexCoord2f(0.0, 1.0);
		glVertex2i(-1, -1);
	glEnd();

	glXSwapBuffers( display, win );
	glXWaitGL();
}

void x_create_glx_window(const char* title, int w, int h) {
	display = XOpenDisplay( NULL );
	const char* xserver = getenv( "DISPLAY" );

	if (display == NULL) {
		printf("Could not establish a connection to X-server '%s'\n", xserver );
		exit(1);
	}

	// query Visual for "TrueColor" and 32 bits depth (RGBA)
	XVisualInfo visualinfo;
	XMatchVisualInfo(display, DefaultScreen(display), 32, TrueColor, &visualinfo);

	// create window
	XSetWindowAttributes attr;
	attr.colormap   = XCreateColormap( display, DefaultRootWindow(display), visualinfo.visual, AllocNone );
	attr.event_mask = ExposureMask | KeyPressMask | StructureNotifyMask;
	attr.background_pixmap = None;
	attr.border_pixel = 0;
	win = XCreateWindow( display, DefaultRootWindow(display),
		50, 300, w, h, // x,y,width,height : are possibly opverwriteen by window manager
		0, visualinfo.depth, InputOutput, visualinfo.visual,
		CWColormap|CWEventMask|CWBackPixmap|CWBorderPixel, &attr);

	// set title bar name of window
	XStoreName( display, win, title );

	// say window manager which position we would prefer
	XSizeHints sizehints;
	sizehints.flags  = PPosition | PSize;
	sizehints.x      = 50;
	sizehints.y      = 300;
	sizehints.width  = w;
	sizehints.height = h;
	XSetWMNormalHints( display, win, &sizehints );

	// Switch On >> If user pressed close key let window manager only send notification >>
	Atom wm_delete_window = XInternAtom( display, "WM_DELETE_WINDOW", 0);
	XSetWMProtocols( display, win, &wm_delete_window, 1);

	// create OpenGL context
	GLXContext glcontext = glXCreateContext( display, &visualinfo, 0, True );
	if (!glcontext) {
		printf("X11 server '%s' does not support OpenGL\n", xserver );
		exit(1);
	}
	glXMakeCurrent( display, win, glcontext );

	// now let the window appear to the user
	XMapWindow( display, win);

	// set undecorated property
	Atom motif_hints = XInternAtom(display,"_MOTIF_WM_HINTS",False);
	long hint_data[5] = { 0x02, 0x00, 0x00, 0x00, 0x00 };
	XChangeProperty(display,win,motif_hints,XA_CARDINAL,32,PropModeReplace, (unsigned char*)hint_data, 5 );

	// set always-on-top property
	Atom wm_state = XInternAtom(display, "_NET_WM_STATE",       False);
	Atom wm_above = XInternAtom(display, "_NET_WM_STATE_ABOVE", False);
	XChangeProperty(display, win, wm_state, XA_ATOM, 32, PropModeReplace, (unsigned char *)(&wm_above), 1 );

	// set an icon
	unsigned long buffer[] = {
		16, 16,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,213893055,2696592058,4072257977,4072192184,2663037626,213893055,0,0,0,0,0,0,0,0,0,0,2880944055,4208449495,4294440951,4294177779,4208186323,2847258037,0,0,0,0,0,0,0,0,0,0,4105417651,4294440951,4293980400,4293519849,4293914607,4105351858,0,0,0,0,0,0,0,0,0,0,4004491183,4293980400,4293454056,4293190884,4293585642,4121865902,0,0,0,0,0,0,0,0,0,0,3450645676,4292796126,4293125091,4292927712,4292467161,3551308972,0,0,0,0,0,0,0,0,0,0,1436919205,4189040559,4292730333,4292598747,4189040559,1571268519,0,0,0,0,0,0,0,0,0,0,0,3349785001,4289572269,4289835441,3618088871,0,0,0,0,0,0,0,0,0,0,0,0,1739632816,4173184445,4189764282,1856941742,0,0,0,0,0,0,0,0,0,79675327,1504488620,3350245552,4172658101,4259965417,4276874219,4155880885,3299913904,1336650667,41975936,0,0,0,0,0,598255784,3853101481,4106404546,4293256677,4294243572,4293914607,4293783021,4293914607,4292796126,4105878202,3752438185,749250728,0,0,0,0,2980357284,4275097808,4293454056,4293059298,4292796126,4292730333,4292664540,4292532954,4292796126,4293125091,4274834636,3231949731,0,0,0,0,3600719518,4292072403,4292138196,4292006610,4292006610,4292006610,4292006610,4292006610,4292006610,4292138196,4292138196,3634273950,0,0,0,0,4002912151,4292072403,4292203989,4292072403,4292006610,4292006610,4292006610,4292006610,4292072403,4292203989,4291940817,3919026071,0,0,0,0,4052717455,4121405351,4290493371,4291282887,4291677645,4291940817,4292006610,4291611852,4291217094,4290493371,4104299170,3851522449,0,0,0,0,311332494,1787398537,3062795918,3666841487,3968633996,4186606218,4186606218,3935145357,3599666830,2945289613,1653246602,244486802,0,0
	};

	Atom wm_icon = XInternAtom(display, "_NET_WM_ICON", False);
	XChangeProperty(display, win, wm_icon, XA_CARDINAL, 32, PropModeReplace, (unsigned char*)buffer, sizeof(buffer)/sizeof(long));
}

int main(int argc, char* argv[]) {
	x_create_glx_window("TransViewer",width,height);

	gl_init_texture();

	cap = cv::VideoCapture(0, cv::CAP_V4L2);
	if (!cap.isOpened()) { printf("unable to open camera\n"); exit(1); }

	cap.set(cv::CAP_PROP_FRAME_WIDTH,  width);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
	cap.set(cv::CAP_PROP_FOURCC, *((uint32_t*)"YUYV"));
	cap.set(cv::CAP_PROP_CONVERT_RGB, false);

	int isUserWantsWindowToClose = 0;
	int isRedraw = 1;

	while (!isUserWantsWindowToClose) {

		while (XPending(display) > 0) {

			// process event
			XEvent event;
			XNextEvent( display, &event);

			switch (event.type) {  // see 'man XAnyEvent' for a list of available events

				case ClientMessage:
					// check if the client message was send by window manager to indicate user wants to close the window
					if( event.xclient.message_type  == XInternAtom( display, "WM_PROTOCOLS", 1) &&
					    event.xclient.data.l[0]  == (long int)XInternAtom( display, "WM_DELETE_WINDOW", 1))
						isUserWantsWindowToClose = 1;
					break;

				case KeyPress: {
					KeySym key = XLookupKeysym(&event.xkey, 0);
					if ((key == XK_Escape) || (key == XK_q))
						isUserWantsWindowToClose = 1;
					break;
				}
				case Expose:
					if (event.xexpose.count == 0)
						isRedraw = 1;
					break;

				case ConfigureNotify:
					glViewport(0,0,event.xconfigure.width,event.xconfigure.height);
					break;

				default:
					// do nothing
					break;
			}
		}

		gl_update_texture();

		if (isRedraw) {
			gl_idle();
			//isRedraw = 0;
		}

		if (x_wait_for_events(10) < 0)
			break;
	}

	XDestroyWindow( display, win );
	XCloseDisplay( display );

	return 0;
}
