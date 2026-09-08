package dev.hoonex.adofai.gamepatcher;

import android.os.Build;

import com.android.apksig.ApkSigner;
import com.android.apksig.ApkVerifier;

import java.io.File;
import java.security.MessageDigest;
import java.security.cert.X509Certificate;
import java.util.Collections;
import java.util.List;

final class SplitSigner {
    static String signAndVerify(File input, File output, SigningIdentity identity) throws Exception {
        File parent = output.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IllegalStateException("could not create signing output directory: " + parent);
        }
        if (output.exists() && !output.delete()) {
            throw new IllegalStateException("could not replace signed output: " + output);
        }

        ApkSigner.SignerConfig config = new ApkSigner.SignerConfig.Builder(
            "adofai-editor", identity.privateKey, identity.certificates
        ).build();
        ApkSigner signer = new ApkSigner.Builder(Collections.singletonList(config))
            .setInputApk(input)
            .setOutputApk(output)
            .setOtherSignersSignaturesPreserved(false)
            // Keep v1/JAR signing in addition to v2/v3 for broad compatibility, but
            // verify against the platform that will actually install this on-device
            // patched split set. ApkVerifier otherwise checks from the APK's own
            // minSdk upward, which can reject an APK for legacy-platform semantics
            // that are irrelevant to this current-device patch flow.
            .setV1SigningEnabled(true)
            .setV2SigningEnabled(true)
            .setV3SigningEnabled(true)
            .setV4SigningEnabled(false)
            .setMinSdkVersion(Math.max(26, Build.VERSION.SDK_INT))
            .build();
        signer.sign();

        int deviceSdk = Build.VERSION.SDK_INT;
        ApkVerifier.Result result = new ApkVerifier.Builder(output)
            .setMinCheckedPlatformVersion(deviceSdk)
            .setMaxCheckedPlatformVersion(deviceSdk)
            .build()
            .verify();
        if (!result.isVerified()) {
            throw new IllegalStateException(
                "apksig verification failed on SDK " + deviceSdk + ": " + describeErrors(result)
            );
        }
        List<X509Certificate> certs = result.getSignerCertificates();
        if (certs.size() != 1) {
            throw new IllegalStateException("unexpected signer count: " + certs.size());
        }
        String digest = hex(MessageDigest.getInstance("SHA-256").digest(certs.get(0).getEncoded()));
        if (!identity.sha256.equals(digest)) {
            throw new IllegalStateException("split signer mismatch: " + digest + " != " + identity.sha256);
        }
        return digest;
    }

    private static String describeErrors(ApkVerifier.Result result) {
        StringBuilder out = new StringBuilder();
        appendIssues(out, "apk", result.getErrors());
        for (ApkVerifier.Result.V1SchemeSignerInfo signer : result.getV1SchemeSigners()) {
            appendIssues(out, "v1:" + signer.getName(), signer.getErrors());
        }
        for (ApkVerifier.Result.V2SchemeSignerInfo signer : result.getV2SchemeSigners()) {
            appendIssues(out, "v2:#" + (signer.getIndex() + 1), signer.getErrors());
        }
        for (ApkVerifier.Result.V3SchemeSignerInfo signer : result.getV3SchemeSigners()) {
            appendIssues(out, "v3:#" + (signer.getIndex() + 1), signer.getErrors());
        }
        if (out.length() == 0) {
            out.append("no surfaced errors; schemes[v1=")
                .append(result.isVerifiedUsingV1Scheme())
                .append(",v2=").append(result.isVerifiedUsingV2Scheme())
                .append(",v3=").append(result.isVerifiedUsingV3Scheme())
                .append(']');
        }
        return out.toString();
    }

    private static void appendIssues(
        StringBuilder out,
        String scope,
        List<ApkVerifier.IssueWithParams> issues
    ) {
        for (ApkVerifier.IssueWithParams issue : issues) {
            if (out.length() > 0) out.append(" | ");
            out.append(scope).append(':').append(issue);
        }
    }

    private static String hex(byte[] value) {
        StringBuilder out = new StringBuilder(value.length * 2);
        for (byte b : value) out.append(String.format(java.util.Locale.US, "%02x", b & 0xff));
        return out.toString();
    }

    private SplitSigner() {}
}
