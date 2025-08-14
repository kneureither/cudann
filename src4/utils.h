#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <string>

#ifndef LOG_LEVEL
#define LOG_LEVEL "INFO"
#endif

inline void logger(const std::string& message, const std::string& level = "INFO", const std::string& file = __FILE__, const int line = __LINE__) {
    // Only show DEBUG messages if LOG_LEVEL is DEBUG
    if (level == "DEBUG" && std::string(LOG_LEVEL) != "DEBUG") {
        return;
    }
    // Only show INFO messages if LOG_LEVEL is INFO or lower
    if (level == "INFO" && std::string(LOG_LEVEL) == "ERROR") {
        return;
    }
    std::string location = "";
    if (file != __FILE__) {
        location = " ( " + file + ":" + std::to_string(line) + ")";
    }
    std::cout << "[" << level << "] " << message << location << std::endl;
}


#endif
