package com.unity3d.player;

import android.app.Activity;
import android.app.Dialog;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.os.Handler;
import android.os.Looper;
import android.system.Os;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.HorizontalScrollView;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.json.JSONTokener;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**
 * Android-native editing shell for modern ADOFAI charts.
 *
 * The 3.3.x Android package still contains the current chart/runtime model but no
 * serialized scnEditor scene. This shell deliberately edits the chart document
 * outside Unity instead of trying to load an editor scene that is not packaged.
 * Unknown root fields and unknown event payloads remain in the JSONObject unless
 * the user explicitly replaces them through the Raw tab.
 */
public final class MobileEditorShell {
    private static final String TAG = "ADOFAI.MobileEditor";
    private static final String LAUNCHER_TAG = "adofai-mobile-editor-launcher";
    private static final long PICKER_TIMEOUT_MS = 120000L;

    private static final Handler MAIN = new Handler(Looper.getMainLooper());

    private static Activity activity;
    private static Dialog dialog;
    private static JSONObject document;
    private static String currentPath;
    private static boolean dirty;
    private static int currentTab;

    private static TextView pathView;
    private static TextView statusView;
    private static FrameLayout contentView;

    private MobileEditorShell() {}

    private static native boolean nativeQueuePreview(String path);

    public static void installLauncher() {
        final Activity resolved = getCurrentActivity();
        if (resolved == null) {
            Log.e(TAG, "Cannot install editor launcher: no foreground Activity");
            return;
        }
        activity = resolved;
        resolved.runOnUiThread(new Runnable() {
            @Override public void run() {
                installLauncherOnUi(resolved);
            }
        });
    }

    private static Activity getCurrentActivity() {
        if (activity != null && !activity.isFinishing()) return activity;
        try {
            Class<?> unityPlayer = Class.forName("com.unity3d.player.UnityPlayer");
            Field field = unityPlayer.getField("currentActivity");
            Activity value = (Activity) field.get(null);
            if (value != null) return value;
        } catch (Throwable error) {
            Log.w(TAG, "UnityPlayer.currentActivity lookup failed", error);
        }
        return FileSelector.context;
    }

    private static void installLauncherOnUi(final Activity owner) {
        View decor = owner.getWindow().getDecorView();
        if (!(decor instanceof ViewGroup)) {
            Log.e(TAG, "Decor view is not a ViewGroup");
            return;
        }
        ViewGroup root = (ViewGroup) decor;
        if (root.findViewWithTag(LAUNCHER_TAG) != null) return;

        Button button = new Button(owner);
        button.setTag(LAUNCHER_TAG);
        button.setText("Editor");
        button.setAllCaps(false);
        button.setTextColor(Color.WHITE);
        button.setTextSize(14f);
        button.setPadding(dp(14), 0, dp(14), 0);
        GradientDrawable background = new GradientDrawable();
        background.setColor(Color.argb(226, 32, 32, 38));
        background.setCornerRadius(dp(16));
        background.setStroke(dp(1), Color.argb(170, 150, 120, 220));
        button.setBackground(background);
        button.setElevation(dp(6));
        button.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View view) {
                openEditor(owner);
            }
        });

        FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(dp(92), dp(48));
        params.gravity = Gravity.TOP | Gravity.END;
        params.topMargin = dp(20);
        params.rightMargin = dp(14);
        root.addView(button, params);
        Log.i(TAG, "Mobile editor launcher installed");
    }

    private static void openEditor(Activity owner) {
        activity = owner;
        if (dialog != null && dialog.isShowing()) return;

        dialog = new Dialog(owner, android.R.style.Theme_Material_NoActionBar);
        dialog.setContentView(buildEditorRoot(owner));
        dialog.setCanceledOnTouchOutside(false);
        dialog.show();
        Window window = dialog.getWindow();
        if (window != null) {
            window.setLayout(WindowManager.LayoutParams.MATCH_PARENT, WindowManager.LayoutParams.MATCH_PARENT);
            window.setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE);
            window.setStatusBarColor(Color.rgb(18, 18, 22));
            window.setNavigationBarColor(Color.rgb(18, 18, 22));
        }
        showTab(currentTab);
    }

    private static View buildEditorRoot(Activity owner) {
        LinearLayout root = new LinearLayout(owner);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.rgb(18, 18, 22));
        root.setPadding(dp(12), dp(10), dp(12), dp(10));

        LinearLayout titleRow = new LinearLayout(owner);
        titleRow.setOrientation(LinearLayout.HORIZONTAL);
        titleRow.setGravity(Gravity.CENTER_VERTICAL);
        TextView title = text("ADOFAI Mobile Editor", 20, Color.WHITE);
        title.setTypeface(null, android.graphics.Typeface.BOLD);
        titleRow.addView(title, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        Button close = actionButton("Close");
        close.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View view) {
                if (dialog != null) dialog.dismiss();
            }
        });
        titleRow.addView(close);
        root.addView(titleRow);

        pathView = text("No chart open", 12, Color.rgb(175, 175, 185));
        pathView.setSingleLine(false);
        root.addView(pathView, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        HorizontalScrollView actionsScroll = new HorizontalScrollView(owner);
        actionsScroll.setHorizontalScrollBarEnabled(false);
        LinearLayout actions = new LinearLayout(owner);
        actions.setOrientation(LinearLayout.HORIZONTAL);
        actions.setPadding(0, dp(6), 0, dp(6));
        actions.addView(makeAction("Open", new View.OnClickListener() {
            @Override public void onClick(View v) { beginOpen(); }
        }));
        actions.addView(makeAction("Save", new View.OnClickListener() {
            @Override public void onClick(View v) { saveCurrent(true); }
        }));
        actions.addView(makeAction("Save As", new View.OnClickListener() {
            @Override public void onClick(View v) { beginSaveAs(); }
        }));
        actions.addView(makeAction("Preview", new View.OnClickListener() {
            @Override public void onClick(View v) { previewCurrent(); }
        }));
        actionsScroll.addView(actions);
        root.addView(actionsScroll);

        HorizontalScrollView tabsScroll = new HorizontalScrollView(owner);
        tabsScroll.setHorizontalScrollBarEnabled(false);
        LinearLayout tabs = new LinearLayout(owner);
        tabs.setOrientation(LinearLayout.HORIZONTAL);
        String[] names = new String[] {"Chart", "Settings", "Events", "Raw"};
        for (int i = 0; i < names.length; i++) {
            final int tab = i;
            Button button = actionButton(names[i]);
            button.setOnClickListener(new View.OnClickListener() {
                @Override public void onClick(View v) { showTab(tab); }
            });
            tabs.addView(button);
        }
        tabsScroll.addView(tabs);
        root.addView(tabsScroll);

        statusView = text("Open a .adofai chart to begin.", 12, Color.rgb(165, 190, 235));
        statusView.setPadding(0, dp(4), 0, dp(6));
        root.addView(statusView);

        contentView = new FrameLayout(owner);
        root.addView(contentView, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
        return root;
    }

    private static View makeAction(String label, View.OnClickListener listener) {
        Button button = actionButton(label);
        button.setOnClickListener(listener);
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, dp(44));
        params.rightMargin = dp(8);
        button.setLayoutParams(params);
        return button;
    }

    private static Button actionButton(String label) {
        Button button = new Button(activity);
        button.setText(label);
        button.setAllCaps(false);
        button.setTextSize(13f);
        button.setTextColor(Color.WHITE);
        button.setMinHeight(dp(42));
        return button;
    }

    private static TextView text(String value, int sp, int color) {
        TextView view = new TextView(activity);
        view.setText(value);
        view.setTextSize(sp);
        view.setTextColor(color);
        return view;
    }

    private static EditText editor(int minLines) {
        EditText edit = new EditText(activity);
        edit.setTextColor(Color.WHITE);
        edit.setHintTextColor(Color.rgb(120, 120, 132));
        edit.setTextSize(14f);
        edit.setMinLines(minLines);
        edit.setGravity(Gravity.TOP | Gravity.START);
        edit.setPadding(dp(10), dp(8), dp(10), dp(8));
        edit.setBackgroundColor(Color.rgb(35, 35, 43));
        return edit;
    }

    private static void showTab(int tab) {
        currentTab = tab;
        if (contentView == null) return;
        contentView.removeAllViews();
        View child;
        if (document == null) {
            TextView empty = text("No chart is open.", 16, Color.rgb(175, 175, 185));
            empty.setGravity(Gravity.CENTER);
            child = empty;
        } else if (tab == 0) {
            child = buildChartTab();
        } else if (tab == 1) {
            child = buildSettingsTab();
        } else if (tab == 2) {
            child = buildEventsTab();
        } else {
            child = buildRawTab();
        }
        contentView.addView(child, new FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
    }

    private static View scroll(View body) {
        ScrollView scroll = new ScrollView(activity);
        scroll.setFillViewport(true);
        scroll.addView(body, new ScrollView.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        return scroll;
    }

    private static LinearLayout column() {
        LinearLayout layout = new LinearLayout(activity);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(0, dp(6), 0, dp(18));
        return layout;
    }

    private static View buildChartTab() {
        final LinearLayout body = column();
        body.addView(text("Geometry", 18, Color.WHITE));
        body.addView(text("Choose the representation to write. Applying one removes the other so the document has one authoritative tile path.", 12, Color.rgb(170, 170, 180)));

        final Spinner mode = new Spinner(activity);
        final ArrayAdapter<String> modeAdapter = new ArrayAdapter<String>(activity, android.R.layout.simple_spinner_item,
                new String[] {"pathData", "angleData"});
        modeAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        mode.setAdapter(modeAdapter);
        boolean angleMode = document.has("angleData") && !document.has("pathData");
        mode.setSelection(angleMode ? 1 : 0);
        body.addView(mode);

        final EditText value = editor(10);
        if (angleMode) value.setText(angleDataText(document.optJSONArray("angleData")));
        else value.setText(document.optString("pathData", ""));
        body.addView(value, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        mode.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                if (position == 0) value.setText(document.optString("pathData", ""));
                else value.setText(angleDataText(document.optJSONArray("angleData")));
            }
            @Override public void onNothingSelected(AdapterView<?> parent) {}
        });

        Button apply = actionButton("Apply geometry");
        apply.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) {
                try {
                    if (mode.getSelectedItemPosition() == 0) {
                        String path = value.getText().toString().trim();
                        if (path.length() == 0) throw new JSONException("pathData cannot be empty");
                        document.put("pathData", path);
                        document.remove("angleData");
                    } else {
                        JSONArray angles = parseAngleData(value.getText().toString());
                        if (angles.length() == 0) throw new JSONException("angleData cannot be empty");
                        document.put("angleData", angles);
                        document.remove("pathData");
                    }
                    markDirty("Geometry updated");
                } catch (Throwable error) {
                    reportError("Geometry update failed", error);
                }
            }
        });
        body.addView(apply);
        return scroll(body);
    }

    private static JSONArray parseAngleData(String text) throws JSONException {
        JSONArray result = new JSONArray();
        String normalized = text.trim();
        if (normalized.startsWith("[")) {
            return new JSONArray(normalized);
        }
        if (normalized.length() == 0) return result;
        String[] parts = normalized.split("[,\\s]+");
        for (String part : parts) {
            if (part.length() == 0) continue;
            if (part.indexOf('.') >= 0 || part.indexOf('e') >= 0 || part.indexOf('E') >= 0) {
                result.put(Double.parseDouble(part));
            } else {
                result.put(Long.parseLong(part));
            }
        }
        return result;
    }

    private static String angleDataText(JSONArray array) {
        if (array == null) return "";
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < array.length(); i++) {
            if (i > 0) out.append(", ");
            out.append(String.valueOf(array.opt(i)));
        }
        return out.toString();
    }

    private static View buildSettingsTab() {
        final LinearLayout body = column();
        body.addView(text("Settings", 18, Color.WHITE));
        final JSONObject settings;
        try {
            JSONObject existing = document.optJSONObject("settings");
            settings = existing != null ? existing : new JSONObject();
            if (existing == null) document.put("settings", settings);
        } catch (JSONException error) {
            reportError("Could not create settings object", error);
            return scroll(body);
        }

        final List<String> keys = sortedKeys(settings);
        final Spinner keySpinner = new Spinner(activity);
        final ArrayAdapter<String> keyAdapter = new ArrayAdapter<String>(activity, android.R.layout.simple_spinner_item, keys);
        keyAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        keySpinner.setAdapter(keyAdapter);
        body.addView(keySpinner);

        final EditText value = editor(6);
        body.addView(value);
        if (!keys.isEmpty()) value.setText(jsonValueText(settings.opt(keys.get(0))));
        keySpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                if (position >= 0 && position < keys.size()) value.setText(jsonValueText(settings.opt(keys.get(position))));
            }
            @Override public void onNothingSelected(AdapterView<?> parent) {}
        });

        Button apply = actionButton("Apply selected setting");
        apply.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) {
                int index = keySpinner.getSelectedItemPosition();
                if (index < 0 || index >= keys.size()) return;
                try {
                    settings.put(keys.get(index), parseJsonValue(value.getText().toString()));
                    markDirty("Setting updated: " + keys.get(index));
                } catch (Throwable error) {
                    reportError("Setting value is not valid JSON", error);
                }
            }
        });
        body.addView(apply);

        Button delete = actionButton("Delete selected setting");
        delete.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) {
                int index = keySpinner.getSelectedItemPosition();
                if (index < 0 || index >= keys.size()) return;
                settings.remove(keys.get(index));
                markDirty("Setting deleted");
                showTab(1);
            }
        });
        body.addView(delete);

        body.addView(text("Add setting", 16, Color.WHITE));
        final EditText newKey = editor(1);
        newKey.setHint("key");
        final EditText newValue = editor(3);
        newValue.setHint("JSON value, e.g. 120 or \"song.ogg\"");
        body.addView(newKey);
        body.addView(newValue);
        Button add = actionButton("Add setting");
        add.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) {
                String key = newKey.getText().toString().trim();
                if (key.length() == 0) return;
                try {
                    settings.put(key, parseJsonValue(newValue.getText().toString()));
                    markDirty("Setting added: " + key);
                    showTab(1);
                } catch (Throwable error) {
                    reportError("New setting value is not valid JSON", error);
                }
            }
        });
        body.addView(add);
        return scroll(body);
    }

    private static View buildEventsTab() {
        final LinearLayout body = column();
        body.addView(text("Actions / Decorations", 18, Color.WHITE));

        final Spinner group = new Spinner(activity);
        ArrayAdapter<String> groups = new ArrayAdapter<String>(activity, android.R.layout.simple_spinner_item,
                new String[] {"actions", "decorations"});
        groups.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        group.setAdapter(groups);
        body.addView(group);

        final ListView list = new ListView(activity);
        list.setBackgroundColor(Color.rgb(28, 28, 34));
        body.addView(list, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(230)));

        final EditText raw = editor(10);
        raw.setHint("Selected event object JSON");
        body.addView(raw);
        final int[] selected = new int[] {-1};

        final Runnable refresh = new Runnable() {
            @Override public void run() {
                String name = group.getSelectedItemPosition() == 1 ? "decorations" : "actions";
                JSONArray array = getOrCreateArray(name);
                List<String> labels = new ArrayList<String>();
                for (int i = 0; i < array.length(); i++) {
                    JSONObject event = array.optJSONObject(i);
                    if (event == null) labels.add(i + ": <non-object>");
                    else labels.add(i + ": " + event.optString("eventType", "<unknown>") + "  floor=" + event.opt("floor"));
                }
                ArrayAdapter<String> adapter = new ArrayAdapter<String>(activity, android.R.layout.simple_list_item_1, labels);
                list.setAdapter(adapter);
                selected[0] = -1;
                raw.setText("");
            }
        };

        group.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(AdapterView<?> parent, View view, int position, long id) { refresh.run(); }
            @Override public void onNothingSelected(AdapterView<?> parent) {}
        });
        list.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                String name = group.getSelectedItemPosition() == 1 ? "decorations" : "actions";
                JSONArray array = getOrCreateArray(name);
                JSONObject event = array.optJSONObject(position);
                selected[0] = position;
                raw.setText(event == null ? String.valueOf(array.opt(position)) : event.toString(2));
            }
        });
        refresh.run();

        Button apply = actionButton("Apply selected object");
        apply.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) {
                if (selected[0] < 0) return;
                try {
                    JSONObject replacement = new JSONObject(sanitizeJson(raw.getText().toString()));
                    String name = group.getSelectedItemPosition() == 1 ? "decorations" : "actions";
                    JSONArray array = getOrCreateArray(name);
                    array.put(selected[0], replacement);
                    markDirty("Event object updated");
                    refresh.run();
                } catch (Throwable error) {
                    reportError("Event object is not valid JSON", error);
                }
            }
        });
        body.addView(apply);

        Button add = actionButton("Add event object");
        add.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) {
                try {
                    boolean decoration = group.getSelectedItemPosition() == 1;
                    JSONObject event = new JSONObject();
                    event.put("floor", 0);
                    if (decoration) {
                        event.put("eventType", "AddDecoration");
                        event.put("decorationImage", "");
                    } else {
                        event.put("eventType", "EditorComment");
                        event.put("comment", "");
                    }
                    getOrCreateArray(decoration ? "decorations" : "actions").put(event);
                    markDirty("Event object added");
                    refresh.run();
                } catch (Throwable error) {
                    reportError("Could not add event", error);
                }
            }
        });
        body.addView(add);

        Button delete = actionButton("Delete selected object");
        delete.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) {
                if (selected[0] < 0) return;
                String name = group.getSelectedItemPosition() == 1 ? "decorations" : "actions";
                getOrCreateArray(name).remove(selected[0]);
                markDirty("Event object deleted");
                refresh.run();
            }
        });
        body.addView(delete);
        return scroll(body);
    }

    private static View buildRawTab() {
        LinearLayout body = column();
        body.addView(text("Raw document", 18, Color.WHITE));
        body.addView(text("Fallback for current/future fields not covered by the structured tabs. Applying replaces the in-memory root only after the entire object parses successfully.", 12, Color.rgb(170, 170, 180)));
        final EditText raw = editor(22);
        raw.setText(document.toString(2));
        body.addView(raw);
        Button apply = actionButton("Apply raw JSON");
        apply.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) {
                try {
                    JSONObject replacement = new JSONObject(sanitizeJson(raw.getText().toString()));
                    document = replacement;
                    markDirty("Raw document applied");
                    showTab(3);
                } catch (Throwable error) {
                    reportError("Raw document is not valid JSON", error);
                }
            }
        });
        body.addView(apply);
        return scroll(body);
    }

    private static JSONArray getOrCreateArray(String key) {
        JSONArray value = document.optJSONArray(key);
        if (value != null) return value;
        value = new JSONArray();
        try { document.put(key, value); }
        catch (JSONException impossible) { throw new IllegalStateException(impossible); }
        return value;
    }

    private static List<String> sortedKeys(JSONObject object) {
        List<String> keys = new ArrayList<String>();
        Iterator<String> iterator = object.keys();
        while (iterator.hasNext()) keys.add(iterator.next());
        Collections.sort(keys);
        return keys;
    }

    private static Object parseJsonValue(String text) throws JSONException {
        String trimmed = text == null ? "" : text.trim();
        if (trimmed.length() == 0) return "";
        return new JSONTokener(trimmed).nextValue();
    }

    private static String jsonValueText(Object value) {
        if (value == null || value == JSONObject.NULL) return "null";
        if (value instanceof JSONObject) return ((JSONObject) value).toString(2);
        if (value instanceof JSONArray) return ((JSONArray) value).toString(2);
        if (value instanceof String) return JSONObject.quote((String) value);
        return String.valueOf(value);
    }

    private static void beginOpen() {
        setStatus("Opening file picker…", false);
        FileSelector.selectFile("adofai");
        awaitPicker(new PickerCompletion() {
            @Override public void complete(String path) {
                if (path.length() == 0) {
                    setStatus("Open cancelled", false);
                    return;
                }
                loadPath(path);
            }
        });
    }

    private static void beginSaveAs() {
        if (document == null) {
            setStatus("No chart to save", true);
            return;
        }
        String name = "level.adofai";
        if (currentPath != null) name = new File(currentPath).getName();
        FileSelector.saveAs(name);
        setStatus("Choose a save path…", false);
        awaitPicker(new PickerCompletion() {
            @Override public void complete(String path) {
                if (path.length() == 0) {
                    setStatus("Save As cancelled", false);
                    return;
                }
                if (!path.toLowerCase().endsWith(".adofai")) path += ".adofai";
                saveToPath(path, true);
            }
        });
    }

    private interface PickerCompletion { void complete(String path); }

    private static void awaitPicker(final PickerCompletion completion) {
        final long deadline = System.currentTimeMillis() + PICKER_TIMEOUT_MS;
        MAIN.post(new Runnable() {
            @Override public void run() {
                if (FileSelector.isDone) {
                    String value = FileSelector.getFilePath();
                    completion.complete(value == null ? "" : value);
                    return;
                }
                if (System.currentTimeMillis() >= deadline) {
                    setStatus("File picker timed out", true);
                    return;
                }
                MAIN.postDelayed(this, 100L);
            }
        });
    }

    private static void loadPath(String path) {
        try {
            String raw = readText(new File(path));
            JSONObject parsed = new JSONObject(sanitizeJson(raw));
            if (!(parsed.opt("settings") instanceof JSONObject)) parsed.put("settings", new JSONObject());
            if (!(parsed.opt("actions") instanceof JSONArray)) parsed.put("actions", new JSONArray());
            if (!(parsed.opt("decorations") instanceof JSONArray)) parsed.put("decorations", new JSONArray());
            document = parsed;
            currentPath = path;
            dirty = false;
            updatePath();
            setStatus("Loaded successfully", false);
            showTab(currentTab);
        } catch (Throwable error) {
            reportError("Could not open chart", error);
        }
    }

    private static String readText(File file) throws Exception {
        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
        StringBuilder out = new StringBuilder((int) Math.min(file.length(), 4 * 1024 * 1024));
        char[] buffer = new char[16384];
        int count;
        while ((count = reader.read(buffer)) != -1) out.append(buffer, 0, count);
        reader.close();
        return out.toString();
    }

    private static String sanitizeJson(String input) {
        if (input == null) return "{}";
        String source = input.length() > 0 && input.charAt(0) == '\ufeff' ? input.substring(1) : input;
        StringBuilder escaped = new StringBuilder(source.length() + 32);
        boolean inString = false;
        boolean slash = false;
        for (int i = 0; i < source.length(); i++) {
            char c = source.charAt(i);
            if (inString) {
                if (slash) {
                    escaped.append(c);
                    slash = false;
                } else if (c == '\\') {
                    escaped.append(c);
                    slash = true;
                } else if (c == '"') {
                    escaped.append(c);
                    inString = false;
                } else if (c < 0x20) {
                    String hex = Integer.toHexString(c);
                    escaped.append("\\u");
                    for (int pad = hex.length(); pad < 4; pad++) escaped.append('0');
                    escaped.append(hex);
                } else {
                    escaped.append(c);
                }
            } else {
                escaped.append(c);
                if (c == '"') inString = true;
            }
        }

        String fixed = escaped.toString();
        StringBuilder out = new StringBuilder(fixed.length());
        inString = false;
        slash = false;
        for (int i = 0; i < fixed.length(); i++) {
            char c = fixed.charAt(i);
            if (inString) {
                out.append(c);
                if (slash) slash = false;
                else if (c == '\\') slash = true;
                else if (c == '"') inString = false;
                continue;
            }
            if (c == '"') {
                inString = true;
                out.append(c);
                continue;
            }
            if (c == ',') {
                int next = i + 1;
                while (next < fixed.length() && Character.isWhitespace(fixed.charAt(next))) next++;
                if (next < fixed.length() && (fixed.charAt(next) == '}' || fixed.charAt(next) == ']')) continue;
            }
            out.append(c);
        }
        return out.toString();
    }

    private static boolean saveCurrent(boolean announce) {
        if (document == null || currentPath == null) {
            if (announce) setStatus("No chart/path to save", true);
            return false;
        }
        return saveToPath(currentPath, announce);
    }

    private static boolean saveToPath(String path, boolean announce) {
        File temp = null;
        try {
            File target = new File(path);
            File parent = target.getAbsoluteFile().getParentFile();
            if (parent == null) throw new IllegalStateException("Target has no parent directory");
            if (!parent.exists() && !parent.mkdirs()) throw new IllegalStateException("Could not create target directory");
            temp = new File(parent, "." + target.getName() + ".mobile-editor." + android.os.Process.myPid() + ".tmp");

            FileOutputStream stream = new FileOutputStream(temp, false);
            OutputStreamWriter writer = new OutputStreamWriter(stream, "UTF-8");
            writer.write(document.toString(2));
            writer.write("\n");
            writer.flush();
            stream.getFD().sync();
            writer.close();

            Os.rename(temp.getAbsolutePath(), target.getAbsolutePath());
            currentPath = target.getAbsolutePath();
            dirty = false;
            updatePath();
            if (announce) setStatus("Saved", false);
            return true;
        } catch (Throwable error) {
            if (temp != null) temp.delete();
            reportError("Save failed", error);
            return false;
        }
    }

    private static void previewCurrent() {
        if (document == null || currentPath == null) {
            setStatus("Open a chart before previewing", true);
            return;
        }
        if (dirty && !saveCurrent(false)) return;
        try {
            if (!nativeQueuePreview(currentPath)) {
                setStatus("3.3 preview bridge is not ready", true);
                return;
            }
            setStatus("Preview queued in the current 3.3 runtime", false);
            if (dialog != null) dialog.dismiss();
        } catch (Throwable error) {
            reportError("Preview request failed", error);
        }
    }

    private static void markDirty(String message) {
        dirty = true;
        updatePath();
        setStatus(message, false);
    }

    private static void updatePath() {
        if (pathView == null) return;
        String value = currentPath == null ? "No chart open" : currentPath;
        if (dirty) value += "  • modified";
        pathView.setText(value);
    }

    private static void setStatus(String message, boolean error) {
        if (statusView != null) {
            statusView.setText(message);
            statusView.setTextColor(error ? Color.rgb(245, 130, 125) : Color.rgb(165, 190, 235));
        }
        if (error && activity != null) Toast.makeText(activity, message, Toast.LENGTH_LONG).show();
    }

    private static void reportError(String prefix, Throwable error) {
        Log.e(TAG, prefix, error);
        setStatus(prefix + ": " + error.getMessage(), true);
    }

    private static int dp(int value) {
        Activity owner = activity != null ? activity : getCurrentActivity();
        if (owner == null) return value;
        return Math.round(value * owner.getResources().getDisplayMetrics().density);
    }
}