package dev.hoonex.adofai.gamepatcher;

import android.security.keystore.KeyGenParameterSpec;
import android.security.keystore.KeyProperties;

import java.math.BigInteger;
import java.security.KeyPairGenerator;
import java.security.KeyStore;
import java.security.MessageDigest;
import java.security.PrivateKey;
import java.security.cert.X509Certificate;
import java.util.Collections;
import java.util.Date;
import java.util.List;

import javax.security.auth.x500.X500Principal;

final class SigningIdentity {
    private static final String STORE = "AndroidKeyStore";
    private static final String ALIAS = "adofai-mobile-editor-sideload-v1";

    final PrivateKey privateKey;
    final X509Certificate certificate;
    final List<X509Certificate> certificates;
    final String sha256;

    private SigningIdentity(PrivateKey privateKey, X509Certificate certificate) throws Exception {
        this.privateKey = privateKey;
        this.certificate = certificate;
        this.certificates = Collections.singletonList(certificate);
        this.sha256 = hex(MessageDigest.getInstance("SHA-256").digest(certificate.getEncoded()));
    }

    static SigningIdentity loadOrCreate() throws Exception {
        KeyStore keyStore = KeyStore.getInstance(STORE);
        keyStore.load(null);
        if (!keyStore.containsAlias(ALIAS)) {
            KeyPairGenerator generator = KeyPairGenerator.getInstance(KeyProperties.KEY_ALGORITHM_RSA, STORE);
            long now = System.currentTimeMillis();
            KeyGenParameterSpec spec = new KeyGenParameterSpec.Builder(
                ALIAS, KeyProperties.PURPOSE_SIGN | KeyProperties.PURPOSE_VERIFY
            )
                .setKeySize(3072)
                .setDigests(KeyProperties.DIGEST_SHA256, KeyProperties.DIGEST_SHA512)
                .setSignaturePaddings(KeyProperties.SIGNATURE_PADDING_RSA_PKCS1)
                .setCertificateSubject(new X500Principal("CN=ADOFAI Mobile Editor Local Sideload,O=Local Device"))
                .setCertificateSerialNumber(BigInteger.valueOf(now).abs())
                .setCertificateNotBefore(new Date(now - 24L * 60L * 60L * 1000L))
                .setCertificateNotAfter(new Date(now + 20L * 365L * 24L * 60L * 60L * 1000L))
                .setUserAuthenticationRequired(false)
                .build();
            generator.initialize(spec);
            generator.generateKeyPair();
            keyStore.load(null);
        }

        PrivateKey key = (PrivateKey) keyStore.getKey(ALIAS, null);
        X509Certificate cert = (X509Certificate) keyStore.getCertificate(ALIAS);
        if (key == null || cert == null) {
            throw new IllegalStateException("AndroidKeyStore signing identity unavailable");
        }
        return new SigningIdentity(key, cert);
    }

    private static String hex(byte[] value) {
        StringBuilder out = new StringBuilder(value.length * 2);
        for (byte b : value) out.append(String.format(java.util.Locale.US, "%02x", b & 0xff));
        return out.toString();
    }
}
