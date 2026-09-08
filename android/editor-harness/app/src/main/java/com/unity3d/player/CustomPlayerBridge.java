package com.unity3d.player;

import android.app.Activity;
import android.content.Intent;

import dev.hoonex.adofai.companion.PlayerActivity;

/** Opens the clean-room playable preview bundled with ADOFAI Custom. */
public final class CustomPlayerBridge {
    private CustomPlayerBridge() {}

    public static boolean open(String path) {
        Activity owner = FileSelector.context;
        if (owner == null || owner.isFinishing() || path == null || path.length() == 0) return false;
        try {
            Intent intent = new Intent(owner, PlayerActivity.class);
            intent.putExtra(PlayerActivity.EXTRA_CHART_PATH, path);
            owner.startActivity(intent);
            return true;
        } catch (Throwable error) {
            return false;
        }
    }
}
