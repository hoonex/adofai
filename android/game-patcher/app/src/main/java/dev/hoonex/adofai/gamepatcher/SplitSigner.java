package dev.hoonex.adofai.gamepatcher;

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
            .setV1SigningEnabled(false)
            .setV2SigningEnabled(true)
            .setV3SigningEnabled(true)
            .setV4SigningEnabled(false)
            .setMinSdkVersion(26)
            .build();
        signer.sign();

        ApkVerifier.Result result = new ApkVerifier.Builder(output).build().verify();
        if (!result.isVerified()) {
            throw new IllegalStateException("apksig verification failed: " + result.getErrors());
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

    private static String hex(byte[] value) {
        StringBuilder out = new StringBuilder(value.length * 2);
        for (byte b : value) out.append(String.format(java.util.Locale.US, "%02x", b & 0xff));
        return out.toString();
    }

    private SplitSigner() {}
}
