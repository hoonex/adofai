#ifndef MOBILE_EDITOR_BRIDGE_H
#define MOBILE_EDITOR_BRIDGE_H

// Installs the small current-runtime hook that drains preview requests on
// Unity's game thread. It fails closed when the 3.3 runtime surface cannot be
// resolved exactly enough for safe invocation.
void InstallMobileEditorPreviewBridge();

#endif // MOBILE_EDITOR_BRIDGE_H
