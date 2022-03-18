#include "utils.h"

#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <regex>
#include <vector>

#include <ctype.h>
#include <unistd.h>


int fourCcFromString(const std::string& in) {
	if (in.empty())
		return 0;

	if (in.size() <= 4)
	{
		// fourcc codes are up to 4 bytes long, right-space-padded and upper-case
		// c.f. http://ffmpeg.org/doxygen/trunk/isom_8c-source.html and
		// c.f. https://www.fourcc.org/codecs.php
		std::array<uint8_t, 4> a = {' ', ' ', ' ', ' '};

		for (size_t i = 0; i < in.size(); ++i)
			a[i] = ::toupper(in[i]);

		return cv::VideoWriter::fourcc(a[0], a[1], a[2], a[3]);
	}
	else if (in.size() == 8)
	{
		// Most people seem to agree on 0x47504A4D being the fourcc code of "MJPG", not the literal translation
		// 0x4D4A5047. This is also what ffmpeg expects.
		return std::stoi(in, nullptr, 16);
	}

	return 0;
}

timestamp_t timestamp() {
	return std::chrono::high_resolution_clock::now();
}
long diffnanosecs(const timestamp_t& t1, const timestamp_t& t2) {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t2).count();
}

bool is_number(const std::string &s) {
	return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
}

std::optional<std::string> resolve_path(const std::string& provided, const std::string& type) {
	std::string result;

	// Check for network (URI) schema and return as-is
	// https://www.rfc-editor.org/rfc/rfc3986#section-3.1
	// however we require at least two chars in the scheme to allow driver letters to work on Windows..
	if (std::regex_match(provided, std::regex("^[[:alpha:]][[:alnum:]+-.]{1,}:.*$")))
		return provided;

	// We use std::ifstream to check we can open each test path read-only, in order:
	// 1. exactly what was provided
	if (std::ifstream(provided).good())
		return provided;

	// to emulate PATH search behaviour (rule of least surprise), we stop here if provided contains path separators
	if (provided.find('/') != provided.npos)
		return {};

	// 2. BACKSCRUB_PATH prefixes if set
	if (getenv("BACKSCRUB_PATH") != nullptr) {
		// getline trick: https://stackoverflow.com/questions/5167625/splitting-a-c-stdstring-using-tokens-e-g
		std::istringstream bsp(getenv("BACKSCRUB_PATH"));

		while (std::getline(bsp, result, ':')) {
			result += "/" + type + "/" + provided;

			if (std::ifstream(result).good())
				return result;
		}
	}

	// 3. XDG standard data location
	result = getenv("XDG_DATA_HOME") ? getenv("XDG_DATA_HOME") : std::string() + getenv("HOME") + "/.local/share";
	result += "/backscrub/" + type + "/" + provided;

	if (std::ifstream(result).good())
		return result;

	// 4. prefixed with compile-time install path
	result = std::string() + _STR(INSTALL_PREFIX) + "/share/backscrub/" + type + "/" + provided;

	if (std::ifstream(result).good())
		return result;

	// 5. relative to current binary location
	// (https://stackoverflow.com/questions/933850/how-do-i-find-the-location-of-the-executable-in-c)
	char binloc[1024];
	ssize_t n = readlink("/proc/self/exe", binloc, sizeof(binloc));

	if (n < 1 || n >= sizeof(binloc))
		return {};

	binloc[n] = 0;

	result = binloc;

	size_t pos = result.rfind('/');

	if (std::string::npos == pos)
		return {};

	pos = result.rfind('/', pos - 1);

	if (std::string::npos == pos)
		return {};

	result.erase(pos);

	result += "/share/backscrub/" + type + "/" + provided;

	if (std::ifstream(result).good())
		return result;

	// development folder?
	result.erase(pos);
	result += "/" + type + "/" + provided;

	if (std::ifstream(result).good())
		return result;

	return {};
}
