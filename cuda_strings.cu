#include "cuda_strings.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>

namespace clover{

std::string matchParam
(FILE * input,
 const char* param_name)
{
    std::string param_string;
    static char name_buf[101];
    rewind(input);
    /* read in line from file */
    while (NULL != fgets(name_buf, 100, input))
    {
        /* ignore line */
        if (NULL != strstr(name_buf, "!"))
        {
            continue;
        }
        /* if it has the parameter name, its the line we want */
        if (NULL != strstr(name_buf, param_name))
        {
            if (NULL != strstr(name_buf, "="))
            {
                *(strstr(name_buf, "=")) = ' ';
                char param_buf[100];
                sscanf(name_buf, "%*s %s", param_buf);
                param_string = std::string(param_buf);
                break;
            }
            else
            {
                param_string = std::string("NO_SETTING");
                break;
            }
        }
    }

    return param_string;
}

bool paramEnabled
(FILE* input, const char* param)
{
    std::string param_string = matchParam(input, param);
    return param_string.size() > 0 ? true : false;
}

int preferredDevice
(FILE* input)
{
    std::string param_string = matchParam(input, "cuda_device");

    int preferred_device;

    if (param_string.size() == 0)
    {
        // not found in file
        preferred_device = -1;
    }
    else
    {
        std::stringstream converter(param_string);

        if (!(converter >> preferred_device))
        {
            preferred_device = -1;
        }
    }

    return preferred_device;
}

}
