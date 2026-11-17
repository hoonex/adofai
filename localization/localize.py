from pathlib import Path
import re, sys, json
root=Path(sys.argv[1])
p=root/'loader/src/overlay.cpp'
s=p.read_text(encoding='utf-8')
mapping=json.loads(Path(__file__).with_name('ko.json').read_text(encoding='utf-8'))
pairs=re.findall(r'(?:S\(\s*|set_status\(\s*[0-9]+\s*,\s*)"([^"]*)"\s*,\s*"([^"]*)"',s)
missing=sorted(set(zh for zh,en in pairs)-mapping.keys())
if missing: raise SystemExit('Untranslated strings: '+repr(missing))
def replace(old,new):
 global s
 if s.count(old)!=1: raise SystemExit('Expected one match: '+old)
 s=s.replace(old,new)
def cpp(v):return json.dumps(v,ensure_ascii=False)
entries='\n'.join('    {'+cpp(k)+', '+cpp(v)+'},' for k,v in mapping.items())
replace('enum Lang { LANG_ZH, LANG_EN };\nstatic Lang g_lang = LANG_ZH;','enum Lang { LANG_ZH, LANG_EN, LANG_KO };\nstatic Lang g_lang = LANG_KO;')
replace('#define S(zh, en) (g_lang == LANG_ZH ? (zh) : (en))', '''static const char* translate_ko(const char* zh, const char* en) {
    static const struct { const char* zh; const char* ko; } strings[] = {
'''+entries+'''
    };
    for (const auto& item : strings) if (strcmp(zh, item.zh) == 0) return item.ko;
    return en;
}
#define S(zh, en) (g_lang == LANG_KO ? translate_ko((zh),(en)) : g_lang == LANG_ZH ? (zh) : (en))''')
replace('    0x3400, 0x4DBF,','    0x1100, 0x11FF,\n    0x3130, 0x318F,\n    0x3400, 0x4DBF,')
replace('    0xFF00, 0xFFEF,','    0xAC00, 0xD7A3,\n    0xFF00, 0xFFEF,')
replace('g_lang = (nf > 0 && strncmp(fields[0], "en", 2) == 0) ? LANG_EN : LANG_ZH;','g_lang = (nf > 0 && strcmp(fields[0], "en") == 0) ? LANG_EN : (nf > 0 && strcmp(fields[0], "zh2") == 0) ? LANG_ZH : LANG_KO;')
replace('g_lang == LANG_EN ? "en" : "zh",','g_lang == LANG_EN ? "en" : g_lang == LANG_ZH ? "zh2" : "ko",')
replace('if (ImGui::Selectable(S("中文", "Chinese"), g_lang == LANG_ZH)) {','if (ImGui::Selectable("한국어", g_lang == LANG_KO)) {\n        g_lang = LANG_KO;\n        save_settings();\n    }\n    if (ImGui::Selectable("中文", g_lang == LANG_ZH)) {')
replace('unsigned char hdr[12];\n    size_t n = fread(hdr, 1, 12, f);','unsigned char hdr[16];\n    size_t n = fread(hdr, 1, 16, f);')
replace('    if (tag != 0x74746366 /*\'ttcf\'*/) return false;','    if (tag != 0x74746366 /*\'ttcf\'*/ || n < 16) return false;')
# Preserve ImGui identity across language changes, keep all labels and popup names as upstream.
p.write_text(s,encoding='utf-8')
print(f'Localized {len(pairs)} UI occurrences, {len(set(x[0] for x in pairs))} unique keys')
