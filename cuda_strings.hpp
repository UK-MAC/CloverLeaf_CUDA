#include <cstdio>
#include <string>

namespace clover
{

/*
 *  Find if tl_use_cg is in the input file
 */
bool paramEnabled
(FILE* input, const char* param);

/*
 *  Returns index of desired device, or -1 if some error occurs (none specified, invalid specification, etc)
 */
int preferredDevice
(FILE* input);

/*
 *  Find out the value of a parameter
 */
std::string matchParam
(FILE * input,
 const char* param_name);

}

