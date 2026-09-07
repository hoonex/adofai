package dev.hoonex.adofai.nativeprobe;

import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.net.Uri;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.List;

/**
 * Read-only resolver probe for possible official-game external .adofai handoff.
 *
 * This does not launch ADOFAI and does not grant a real file. It only asks
 * PackageManager whether the exact installed package advertises an exported
 * ACTION_VIEW activity for representative file/content URI + MIME combinations.
 */
final class IntentCapabilityProbe {
    private IntentCapabilityProbe() {}

    static JSONObject build(PackageManager pm, String targetPackage) throws Exception {
        JSONObject result = new JSONObject();
        JSONArray probes = new JSONArray();
        boolean anyTargetHandler = false;

        anyTargetHandler |= addViewProbe(
                probes,
                pm,
                targetPackage,
                "file-no-mime",
                Uri.parse("file:///sdcard/Download/adofai-intent-probe.adofai"),
                null);
        anyTargetHandler |= addViewProbe(
                probes,
                pm,
                targetPackage,
                "file-application-json",
                Uri.parse("file:///sdcard/Download/adofai-intent-probe.adofai"),
                "application/json");
        anyTargetHandler |= addViewProbe(
                probes,
                pm,
                targetPackage,
                "file-text-plain",
                Uri.parse("file:///sdcard/Download/adofai-intent-probe.adofai"),
                "text/plain");
        anyTargetHandler |= addViewProbe(
                probes,
                pm,
                targetPackage,
                "content-application-json",
                Uri.parse("content://dev.hoonex.adofai.nativeprobe/adofai-intent-probe.adofai"),
                "application/json");
        anyTargetHandler |= addViewProbe(
                probes,
                pm,
                targetPackage,
                "content-octet-stream",
                Uri.parse("content://dev.hoonex.adofai.nativeprobe/adofai-intent-probe.adofai"),
                "application/octet-stream");

        result.put("official_view_handler_detected", anyTargetHandler);
        result.put("probes", probes);
        result.put(
                "interpretation",
                "A matching exported ACTION_VIEW handler is only evidence that Android can route the intent. " +
                "It does not prove the game consumes the URI as a custom level until a separate launch test succeeds.");
        return result;
    }

    private static boolean addViewProbe(
            JSONArray out,
            PackageManager pm,
            String targetPackage,
            String label,
            Uri uri,
            String mime) throws Exception {
        Intent intent = new Intent(Intent.ACTION_VIEW);
        intent.addCategory(Intent.CATEGORY_DEFAULT);
        intent.setPackage(targetPackage);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        if (mime == null) intent.setData(uri);
        else intent.setDataAndType(uri, mime);

        List<ResolveInfo> matches = pm.queryIntentActivities(intent, PackageManager.MATCH_DEFAULT_ONLY);
        JSONArray handlers = new JSONArray();
        boolean anyExported = false;
        for (ResolveInfo match : matches) {
            ActivityInfo activity = match.activityInfo;
            if (activity == null) continue;
            JSONObject handler = new JSONObject();
            handler.put("name", activity.name);
            handler.put("package", activity.packageName);
            handler.put("exported", activity.exported);
            handlers.put(handler);
            if (targetPackage.equals(activity.packageName) && activity.exported) anyExported = true;
        }

        JSONObject item = new JSONObject();
        item.put("label", label);
        item.put("action", Intent.ACTION_VIEW);
        item.put("uri", uri.toString());
        item.put("mime", mime == null ? JSONObject.NULL : mime);
        item.put("handler_count", handlers.length());
        item.put("handlers", handlers);
        item.put("target_exported_handler", anyExported);
        out.put(item);
        return anyExported;
    }
}
