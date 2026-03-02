#ifndef WTC_RUNTIME_POSTBUFFERIZEFIXUPS_H
#define WTC_RUNTIME_POSTBUFFERIZEFIXUPS_H

#include <string>

namespace wtc::runtime {

bool validatePostbufferizeArtifacts(const std::string &artifactDir,
                                    std::string &diagnostic);

// 兼容旧名，语义上等价于 validatePostbufferizeArtifacts。
inline bool applyPostbufferizeFixups(const std::string &artifactDir,
                                     std::string &diagnostic) {
  return validatePostbufferizeArtifacts(artifactDir, diagnostic);
}

} // namespace wtc::runtime

#endif
