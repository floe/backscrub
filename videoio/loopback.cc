/* This is licensed software, @see LICENSE file.
 * Authors - @see AUTHORS file. */

#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>

#include "loopback.h"

void print_format(struct v4l2_format*vid_format) {
	printf("vid_format->type                = %d\n",	vid_format->type );
	printf("vid_format->fmt.pix.width       = %d\n",	vid_format->fmt.pix.width );
	printf("vid_format->fmt.pix.height      = %d\n",	vid_format->fmt.pix.height );
	printf("vid_format->fmt.pix.pixelformat = %d\n",	vid_format->fmt.pix.pixelformat);
	printf("vid_format->fmt.pix.sizeimage   = %d\n",	vid_format->fmt.pix.sizeimage );
	printf("vid_format->fmt.pix.field       = %d\n",	vid_format->fmt.pix.field );
	printf("vid_format->fmt.pix.bytesperline= %d\n",	vid_format->fmt.pix.bytesperline );
	printf("vid_format->fmt.pix.colorspace  = %d\n",	vid_format->fmt.pix.colorspace );
	printf("\n");
}

int loopback_init(const char* device, int w, int h, int debug) {

	struct v4l2_capability vid_caps;
	struct v4l2_format vid_format;

	size_t linewidth = w * 2;
	size_t framesize = h * linewidth;

	int fdwr = 0;
	int ret_code = 0;

	fdwr = open(device, O_RDWR);
	if(fdwr < 0) {
		fprintf(stderr, "%s:%d(%s): Failed to open video device: %s\n", __FILE__, __LINE__, __func__, strerror(errno));
		return -1;
	}

	ret_code = ioctl(fdwr, VIDIOC_QUERYCAP, &vid_caps);
	if(ret_code < 0) {
		fprintf(stderr, "%s:%d(%s): Failed to query device capabilities: %s\n", __FILE__, __LINE__, __func__, strerror(errno));
		close(fdwr);
		return -1;
	}

	memset(&vid_format, 0, sizeof(vid_format));
	//usleep(100000);

	ret_code = ioctl(fdwr, VIDIOC_G_FMT, &vid_format);

	vid_format.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
	vid_format.fmt.pix.width = w;
	vid_format.fmt.pix.height = h;
	vid_format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
	vid_format.fmt.pix.sizeimage = framesize;
	vid_format.fmt.pix.field = V4L2_FIELD_NONE;
	vid_format.fmt.pix.bytesperline = linewidth;
	vid_format.fmt.pix.colorspace = V4L2_COLORSPACE_SRGB;

	ret_code = ioctl(fdwr, VIDIOC_S_FMT, &vid_format);
	if(ret_code < 0) {
		fprintf(stderr, "%s:%d(%s): Failed to set device video format: %s\n", __FILE__, __LINE__, __func__, strerror(errno));
		close(fdwr);
		return -1;
	}

	if (debug) print_format(&vid_format);

	return fdwr;
}

#ifdef standalone

#define FRAME_WIDTH 640
#define FRAME_HEIGHT 480

int main(int argc, char* argv[]) {

	char* video_device = "/dev/video1";

	size_t linewidth = FRAME_WIDTH  * 2;
	size_t framesize = FRAME_HEIGHT * linewidth;

	if(argc>1) {
		video_device=argv[1];
		printf("using output device: %s\n", video_device);
	}

	int fdwr = loopback_init(video_device,FRAME_WIDTH,FRAME_HEIGHT);
	if(fdwr < 0) {
		fprintf(stderr, "Failed to initialize output device %s\n", video_device);
		exit(1);
	}

	uint8_t* buffer = (uint8_t*)malloc(framesize);

	while (true) {
		write(fdwr, buffer, framesize);
		usleep(100000);
		uint64_t* front = (uint64_t*)(buffer);
		*front += 12345;
	}

	pause();

	close(fdwr);

	free(buffer);

	return 0;
}

#endif
