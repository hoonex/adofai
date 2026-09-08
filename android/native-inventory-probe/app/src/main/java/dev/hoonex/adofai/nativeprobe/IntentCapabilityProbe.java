package dev.hoonex.adofai.nativeprobe;

import android.content.ComponentName;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.net.Uri;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.List;

/**
 * Read-only resolver probe for possible official-game external .adofai handoff.
 *
 * This does not launch ADOFAI and does not grant a real file. It asks
 * PackageManager which exported activities the exact installed package exposes
 * and whether representative VIEW/SEND intents can be routed to them.
 */
final class IntentCapabilityProbe {
    private IntentCapabilityProbe() {}

    static JSONObject build(PackageManager pm, String targetPackage) throws Exception {
        JSONObject result = new JSONObject();
        JSONArray probes = new JSONArray();
        boolean anyTargetHandler = false;

        PackageInfo packageInfo = pm.getPackageInfo(targetPackage, PackageManager.GET_ACTIVITIES);
        JSONArray exportedActivities = new JSONArray();
        if (packageInfo.activities != null) {
            for (ActivityInfo activity : packageInfo.activities) {
                if (activity == null || !activity.exported) continue;
                JSONObject item = new JSONObject();
                item.put("name", activity.name);
                item.put("package", activity.packageName);
                item.put("enabled", activity.enabled);
                item.put("permission", activity.permission == null ? JSONObject.NULL : activity.permission);
                exportedActivities.put(item);
            }
        }
        result.put("exported_activities", exportedActivities);

        Intent launchIntent = pm.getLaunchIntentForPackage(targetPackage);
        JSONObject launcher = new JSONObject();
        if (launchIntent != null && launchIntent.getComponent() != null) {
            ComponentName component = launchIntent.getComponent();
            launcher.put("package", component.getPackageName());
            launcher.put("class", component.getClassName());
        } else {
            launcher.put("package", JSONObject.NULL);
            launcher.put("class", JSONObject.NULL);
        }
        result.put("launcher_activity", launcher);

        Uri fileChart = Uri.parse("file:///sdcard/Download/adofai-intent-probe.adofai");
        Uri contentChart = Uri.parse("content://dev.hoonex.adofai.nativeprobe/adofai-intent-probe.adofai");
        Uri httpsChart = Uri.parse("https://example.invalid/adofai-intent-probe.adofai");
        Uri httpsZip = Uri.parse("https://example.invalid/adofai-intent-probe.zip");

        anyTargetHandler |= addViewProbe(probes, pm, targetPackage, "view-file-no-mime", fileChart, null);
        anyTargetHandler |= addViewProbe(probes, pm, targetPackage, "view-file-application-json", fileChart, "application/json");
        anyTargetHandler |= addViewProbe(probes, pm, targetPackage, "view-file-text-plain", fileChart, "text/plain");
        anyTargetHandler |= addViewProbe(probes, pm, targetPackage, "view-content-no-mime", contentChart, null);
        anyTargetHandler |= addViewProbe(probes, pm, targetPackage, "view-content-application-json", contentChart, "application/json");
        anyTargetHandler |= addViewProbe(probes, pm, targetPackage, "view-content-octet-stream", contentChart, "application/octet-stream");
        anyTargetHandler |= addViewProbe(probes, pm, targetPackage, "view-content-text-plain", contentChart, "text/plain");
        anyTargetHandler |= addViewProbe(probes, pm, targetPackage, "view-https-adofai", httpsChart, null);
        anyTargetHandler |= addViewProbe(probes, pm, targetPackage, "view-https-zip", httpsZip, null);

        anyTargetHandler |= addSendStreamProbe(probes, pm, targetPackage, "send-stream-application-json", contentChart, "application/json");
        anyTargetHandler |= addSendStreamProbe(probes, pm, targetPackage, "send-stream-octet-stream", contentChart, "application/octet-stream");
        anyTargetHandler |= addSendStreamProbe(probes, pm, targetPackage, "send-stream-text-plain", contentChart, "text/plain");
        anyTargetHandler |= addSendTextProbe(probes, pm, targetPackage, "send-url-text-plain", httpsChart.toString());

        result.put("official_external_handler_detected", anyTargetHandler);
        result.put("probes", probes);
        result.put(
                "interpretation",
                "A matching exported VIEW/SEND handler is only evidence that Android can route the intent. " +
                "It does not prove the game consumes the URI as a custom level until a separate launch test succeeds. " +
                "No matching handler is strong evidence that a normal non-root cross-app handoff is unavailable through these public intent shapes.");
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
        return addProbe(out, pm, targetPackage, label, intent, uri.toString(), mime);
    }

    private static boolean addSendStreamProbe(
            JSONArray out,
            PackageManager pm,
            String targetPackage,
            String label,
            Uri uri,
            String mime) throws Exception {
        Intent intent = new Intent(Intent.ACTION_SEND);
        intent.addCategory(Intent.CATEGORY_DEFAULT);
        intent.setPackage(targetPackage);
        intent.setType(mime);
        intent.putExtra(Intent.EXTRA_STREAM, uri);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        return addProbe(out, pm, targetPackage, label, intent, uri.toString(), mime);
    }

    private static boolean addSendTextProbe(
            JSONArray out,
            PackageManager pm,
            String targetPackage,
            String label,
            String text) throws Exception {
        Intent intent = new Intent(Intent.ACTION_SEND);
        intent.addCategory(Intent.CATEGORY_DEFAULT);
        intent.setPackage(targetPackage);
        intent.setType("text/plain");
        intent.putExtra(Intent.EXTRA_TEXT, text);
        return addProbe(out, pm, targetPackage, label, intent, text, "text/plain");
    }

    private static boolean addProbe(
            JSONArray out,
            PackageManager pm,
            String targetPackage,
            String label,
            Intent intent,
            String payload,
            String mime) throws Exception {
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
            handler.put("enabled", activity.enabled);
            handlers.put(handler);
            if (targetPackage.equals(activity.packageName) && activity.exported && activity.enabled) anyExported = true;
        }

        JSONObject item = new JSONObject();
        item.put("label", label);
        item.put("action", intent.getAction());
        item.put("payload", payload);
        item.put("mime", mime == null ? JSONObject.NULL : mime);
        item.put("handler_count", handlers.length());
        item.put("handlers", handlers);
        item.put("target_exported_handler", anyExported);
        out.put(item);
        return anyExported;
    }
}
