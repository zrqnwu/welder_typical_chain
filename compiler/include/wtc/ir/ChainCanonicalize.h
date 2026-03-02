#ifndef WTC_IR_CHAINCANONICALIZE_H
#define WTC_IR_CHAINCANONICALIZE_H

#include <string>

namespace wtc::ir {

bool canonicalizeMatmulSoftmaxChain(const std::string &inputPath,
                                    const std::string &outputPath,
                                    std::string &diagnostic);

} // namespace wtc::ir

#endif
