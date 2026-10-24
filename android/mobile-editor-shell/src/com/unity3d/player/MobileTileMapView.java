package com.unity3d.player;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.View;
import android.view.ViewConfiguration;

import org.json.JSONArray;
import org.json.JSONObject;

/**
 * Touch-native floor selection for the 3.3.1 editor shell.
 *
 * This is a schematic chart-path view, not a replacement for Unity's track renderer.
 * In particular, animated MoveTrack/decoration effects are not applied. Selecting a
 * floor never changes the chart; only explicit editor actions can mutate the JSON.
 */
public final class MobileTileMapView extends View {
    public interface FloorListener {
        void onFloorSelected(int floor);
    }

    private static final int MAX_DISPLAYED_FLOORS = 20000;
    private static final String DIRECTIONS = "RpJEToUqGQHWLxNZFVDYBCMA";
    private static final String OFFSETS = "56789hjty";
    private static final double[] OFFSET_DEGREES = {72, -72, 52, -52, -30, 120, -120, 60, 300};

    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final ScaleGestureDetector scaleDetector;
    private final int touchSlop;
    private final float density;
    private final float[] tileX;
    private final float[] tileY;
    private final FloorListener listener;
    private final float minX;
    private final float maxX;
    private final float minY;
    private final float maxY;

    private float zoom = 1f;
    private float offsetX;
    private float offsetY;
    private int selectedFloor;
    private float downX;
    private float downY;
    private float lastX;
    private float lastY;
    private boolean moved;
    private boolean multiTouch;
    private boolean fitted;

    public MobileTileMapView(Context context, JSONObject chart, int initialFloor, FloorListener callback) {
        super(context);
        listener = callback;
        density = context.getResources().getDisplayMetrics().density;
        touchSlop = ViewConfiguration.get(context).getScaledTouchSlop();
        final double[] directions = decodeAngles(chart);
        if (directions.length == 0) throw new IllegalArgumentException("Chart has no floors");
        tileX = new float[directions.length];
        tileY = new float[directions.length];
        double x = 0d;
        double y = 0d;
        double lowX = 0d, highX = 0d, lowY = 0d, highY = 0d;
        JSONObject[] positionOffsets = positionOffsets(chart, directions.length);
        for (int i = 0; i < directions.length; i++) {
            JSONObject move = positionOffsets[i];
            if (move != null) {
                JSONArray offset = move.optJSONArray("positionOffset");
                if (offset != null && offset.length() >= 2) {
                    Object ox = offset.opt(0);
                    Object oy = offset.opt(1);
                    if (!(ox instanceof Number) || !(oy instanceof Number)) {
                        throw new IllegalArgumentException("Invalid PositionTrack on floor " + i);
                    }
                    double dx = ((Number) ox).doubleValue();
                    double dy = ((Number) oy).doubleValue();
                    if (!Double.isFinite(dx) || !Double.isFinite(dy)) {
                        throw new IllegalArgumentException("Non-finite PositionTrack on floor " + i);
                    }
                    x += dx;
                    y += dy;
                }
            }
            if (!Double.isFinite(x) || !Double.isFinite(y)
                    || Math.abs(x) > 1000000d || Math.abs(y) > 1000000d) {
                throw new IllegalArgumentException("Chart position exceeds safe preview range");
            }
            tileX[i] = (float) x;
            tileY[i] = (float) y;
            lowX = Math.min(lowX, x);
            highX = Math.max(highX, x);
            lowY = Math.min(lowY, y);
            highY = Math.max(highY, y);
            double radians = Math.toRadians(directions[i]);
            x += Math.cos(radians);
            y += Math.sin(radians);
        }
        minX = (float) lowX;
        maxX = (float) highX;
        minY = (float) lowY;
        maxY = (float) highY;
        selectedFloor = Math.max(0, Math.min(initialFloor, tileX.length - 1));
        scaleDetector = new ScaleGestureDetector(context,
                new ScaleGestureDetector.SimpleOnScaleGestureListener() {
                    @Override public boolean onScale(ScaleGestureDetector detector) {
                        final float factor = detector.getScaleFactor();
                        if (!Float.isFinite(factor) || factor <= 0f) return false;
                        float next = Math.max(2f * density, Math.min(450f * density, zoom * factor));
                        float focusX = detector.getFocusX();
                        float focusY = detector.getFocusY();
                        float ratio = next / zoom;
                        offsetX = focusX - (focusX - offsetX) * ratio;
                        offsetY = focusY - (focusY - offsetY) * ratio;
                        zoom = next;
                        invalidate();
                        return true;
                    }
                });
        setContentDescription("Chart floor map. Tap a floor, drag to pan, pinch to zoom.");
    }

    public int floorCount() {
        return tileX.length;
    }

    public void selectFloor(int floor, boolean center) {
        selectedFloor = Math.max(0, Math.min(floor, tileX.length - 1));
        if (center && getWidth() > 0 && getHeight() > 0) {
            offsetX = getWidth() / 2f - tileX[selectedFloor] * zoom;
            offsetY = getHeight() / 2f + tileY[selectedFloor] * zoom;
        }
        invalidate();
    }

    private static JSONObject[] positionOffsets(JSONObject chart, int count) {
        JSONObject[] result = new JSONObject[count];
        JSONArray actions = chart.optJSONArray("actions");
        if (actions == null) return result;
        for (int i = 0; i < actions.length(); i++) {
            JSONObject action = actions.optJSONObject(i);
            if (action == null || !"PositionTrack".equals(action.optString("eventType"))) continue;
            Object editorOnly = action.opt("editorOnly");
            if (Boolean.TRUE.equals(editorOnly) || "Enabled".equals(editorOnly)) continue;
            int floor = action.optInt("floor", -1);
            if (floor >= 0 && floor < count && action.optJSONArray("positionOffset") != null) {
                result[floor] = action;
            }
        }
        return result;
    }

    private static double[] decodeAngles(JSONObject chart) {
        if (chart.has("pathData")) {
            String path = chart.optString("pathData", "");
            guardFloorCount(path.length());
            double[] result = new double[path.length()];
            double last = 0d;
            for (int i = 0; i < path.length(); i++) {
                char symbol = path.charAt(i);
                int absolute = DIRECTIONS.indexOf(symbol);
                int relative = OFFSETS.indexOf(symbol);
                if (symbol == '!') last = 999d;
                else if (absolute >= 0) last = absolute * 15d;
                else if (relative >= 0) last += OFFSET_DEGREES[relative];
                result[i] = last;
            }
            return resolveMidspins(result);
        }
        JSONArray raw = chart.optJSONArray("angleData");
        if (raw == null) return new double[0];
        guardFloorCount(raw.length());
        double[] result = new double[raw.length()];
        for (int i = 0; i < raw.length(); i++) {
            Object source = raw.opt(i);
            if (!(source instanceof Number)) {
                throw new IllegalArgumentException("angleData has a non-numeric floor at " + i);
            }
            result[i] = ((Number) source).doubleValue();
            if (!Double.isFinite(result[i])) {
                throw new IllegalArgumentException("angleData has a non-finite floor at " + i);
            }
            double offset = result[i] == 555d ? 72d : result[i] == 666d ? -72d
                    : result[i] == 777d ? 52d : result[i] == 888d ? -52d : Double.NaN;
            if (!Double.isNaN(offset)) result[i] = (i == 0 ? 0d : result[i - 1]) + offset;
        }
        return resolveMidspins(result);
    }

    private static void guardFloorCount(int count) {
        if (count > MAX_DISPLAYED_FLOORS) {
            throw new IllegalArgumentException("Map view limited to " + MAX_DISPLAYED_FLOORS
                    + " floors; the original chart is unchanged");
        }
    }

    private static double[] resolveMidspins(double[] angles) {
        double[] raw = angles.clone();
        for (int i = 0; i < angles.length; i++) {
            if (raw[i] != 999d) continue;
            int back = i - 1;
            // Preserve sequential midspin direction changes as the pinned ADOFAI-JS
            // chart geometry calculation does, rather than treating 999 as radians.
            while (back >= 0 && raw[back] == 999d) back--;
            double base = back >= 0 ? raw[back] : 0d;
            angles[i] = base + Math.max(0, i - back - 1) * 180d;
        }
        return angles;
    }

    @Override protected void onSizeChanged(int w, int h, int oldW, int oldH) {
        super.onSizeChanged(w, h, oldW, oldH);
        if (!fitted && w > 0 && h > 0) {
            float padding = 48f * density;
            float spanX = Math.max(1f, maxX - minX + 1f);
            float spanY = Math.max(1f, maxY - minY + 1f);
            zoom = Math.max(2f * density, Math.min(72f * density,
                    Math.min((w - 2f * padding) / spanX, (h - 2f * padding) / spanY)));
            offsetX = w / 2f - (maxX + minX) / 2f * zoom;
            offsetY = h / 2f + (maxY + minY) / 2f * zoom;
            fitted = true;
        }
    }

    @Override protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        canvas.drawColor(Color.rgb(25, 26, 33));
        paint.setStrokeWidth(3f * density);
        paint.setColor(Color.rgb(99, 110, 134));
        paint.setStyle(Paint.Style.STROKE);
        for (int i = 1; i < tileX.length; i++) {
            float x1 = sx(tileX[i - 1]), y1 = sy(tileY[i - 1]);
            float x2 = sx(tileX[i]), y2 = sy(tileY[i]);
            if (visible(x1, y1) || visible(x2, y2)) canvas.drawLine(x1, y1, x2, y2, paint);
        }
        float radius = Math.max(3f * density, Math.min(18f * density, zoom * 0.33f));
        paint.setStyle(Paint.Style.FILL);
        for (int i = 0; i < tileX.length; i++) {
            float x = sx(tileX[i]), y = sy(tileY[i]);
            if (!visible(x, y)) continue;
            paint.setColor(i == selectedFloor ? Color.rgb(246, 202, 96)
                    : i == 0 ? Color.rgb(118, 213, 182) : Color.rgb(183, 172, 227));
            canvas.drawCircle(x, y, radius, paint);
        }
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(3f * density);
        paint.setColor(Color.WHITE);
        canvas.drawCircle(sx(tileX[selectedFloor]), sy(tileY[selectedFloor]), radius + 4f * density, paint);
        paint.setStyle(Paint.Style.FILL);
    }

    private float sx(float x) { return offsetX + x * zoom; }
    private float sy(float y) { return offsetY - y * zoom; }
    private boolean visible(float x, float y) {
        return x >= -40f * density && x <= getWidth() + 40f * density
                && y >= -40f * density && y <= getHeight() + 40f * density;
    }

    @Override public boolean onTouchEvent(MotionEvent event) {
        scaleDetector.onTouchEvent(event);
        switch (event.getActionMasked()) {
            case MotionEvent.ACTION_DOWN:
                downX = lastX = event.getX();
                downY = lastY = event.getY();
                moved = false;
                multiTouch = false;
                return true;
            case MotionEvent.ACTION_POINTER_DOWN:
                multiTouch = true;
                moved = true;
                return true;
            case MotionEvent.ACTION_POINTER_UP:
                // Reset the pan anchor to the surviving finger so the next
                // single-pointer MOVE cannot jump to an old touch coordinate.
                int survivor = event.getActionIndex() == 0 ? 1 : 0;
                if (survivor < event.getPointerCount()) {
                    lastX = event.getX(survivor);
                    lastY = event.getY(survivor);
                }
                multiTouch = true;
                moved = true;
                return true;
            case MotionEvent.ACTION_MOVE:
                if (event.getPointerCount() == 1 && !scaleDetector.isInProgress()) {
                    float x = event.getX(), y = event.getY();
                    if (Math.hypot(x - downX, y - downY) > touchSlop) moved = true;
                    if (moved) {
                        offsetX += x - lastX;
                        offsetY += y - lastY;
                        invalidate();
                    }
                    lastX = x; lastY = y;
                }
                return true;
            case MotionEvent.ACTION_UP:
                if (!moved && !multiTouch
                        && Math.hypot(event.getX() - downX, event.getY() - downY) <= touchSlop) {
                    int nearest = -1;
                    double distance2 = Double.POSITIVE_INFINITY;
                    for (int i = 0; i < tileX.length; i++) {
                        double dx = sx(tileX[i]) - event.getX();
                        double dy = sy(tileY[i]) - event.getY();
                        double d2 = dx * dx + dy * dy;
                        if (d2 < distance2) { nearest = i; distance2 = d2; }
                    }
                    // Standard Android 48dp touch target, restricted to the nearest floor.
                    if (nearest >= 0 && distance2 <= (24f * density) * (24f * density)) {
                        selectFloor(nearest, false);
                        performClick();
                        if (listener != null) listener.onFloorSelected(nearest);
                    }
                }
                return true;
            case MotionEvent.ACTION_CANCEL:
                multiTouch = true;
                return true;
            default:
                return true;
        }
    }

    @Override public boolean performClick() {
        super.performClick();
        return true;
    }
}
