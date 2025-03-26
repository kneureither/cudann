#ifndef UTILS_H
#define UTILS_H

#include <iostream>


inline void logger(const std::string& message, const std::string& level = "INFO", const std::string& file = __FILE__, const int line = __LINE__) {
    std::cout << "[" << level << "] " << message << " (" << file << ":" << line << ")" << std::endl;
}


#endif
