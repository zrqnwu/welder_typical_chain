#ifndef WTC_IR_TAGGING_H
#define WTC_IR_TAGGING_H

#include <string>

namespace wtc::ir {

bool tagTypicalChainOps(const std::string &inputPath,
                        const std::string &outputPath,
                        const std::string &tagsJsonPath,
                        std::string &diagnostic);

} // namespace wtc::ir

#endif
