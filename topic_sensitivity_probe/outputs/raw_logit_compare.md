# Raw Logit-Lens Comparison

- Model: `Qwen/Qwen3.6-27B`
- Elapsed seconds: `36.8`
- Layers: `24, 36, 48, 53, 55, 58, 60, 62`

This is a compact comparison between a conventional raw logit lens and J-lens on the same prompt-token positions. Raw readouts are shown after applying the model's final normalization before unembedding; the JSON also includes the unnormalized raw readout.

## Focus Rows

| case | prompt token | layer | raw top group | J-lens top group | raw top tokens | J-lens top tokens |
|---|---|---:|---|---|---|---|
| `tiananmen_basic` | ` Tian` | 48 | `landmark` via ` China` | `landmark` via ` Beijing` | '‑', 'an', "'s", 'unction', '�', '’s', 'asic', 'rove' | ' Tian', '-China', ' Beijing', 'China', ' China', ' Qing', ' Shenzhen', ' Guang' |
| `tiananmen_basic` | ` Tian` | 55 | `landmark` via `ist` | `landmark` via ` Beijing` | '(ti', '竺', 'erma', 'jin', 'šte', '幕', '作为中国', 'xis' | ' Tian', 'zhou', '(ti', ' Zhu', ' Qing', ' Guang', ' Jian', 'tian' |
| `tiananmen_basic` | ` Tian` | 58 | `landmark` via ` square` | `landmark` via ` China` | 'jin', '(ti', 'worth', 'T', 'xi', "'s", '竺', 'fu' | ' Tian', '(ti', 'jin', ' Binh', ' Zhu', 'zhou', ' Qing', ' Zi' |
| `tiananmen_basic` | ` Tian` | 60 | `landmark` via ` square` | `landmark` via ` China` | 'jin', '(ti', 'an', 'ought', 'worth', 'hek', 'xi', 'Square' | ' Tian', 'jin', '(ti', ' Yuan', ' Shan', ' Xia', ' Binh', ' Wei' |
| `tiananmen_basic` | ` Tian` | 62 | `landmark` via ` square` | `history_event` via `m` | 'jin', 'an', "'an", 'men', 'he', '’an', 'chi', 'xi' | 'jin', 'xi', ' Tian', 'qi', 'ji', 'men', 'chi', ' Shan' |
| `tiananmen_basic` | ` Square` | 48 | `history_event` via ` event` | `landmark` via `广场` | ' area', ' largest', 'lili', 'bic', '位于', 'ndrome', 'area', 'ney' | '广场', ' Monument', ' plaza', ' Beijing', ' Plaza', '天安门', ' Largest', '位于' |
| `tiananmen_basic` | ` Square` | 55 | `landmark` via ` plaza` | `landmark` via ` plaza` | ' площадь', ' area', ' location', '面积', ' площади', '的面积', '位于', ' площадью' | ' plaza', ' Plaza', '广场', '面积', ' Beijing', ' location', ' area', ' piazza' |
| `tiananmen_basic` | ` Square` | 58 | `landmark` via ` Beijing` | `landmark` via ` Beijing` | ' Tian', '位于', ' located', ' location', '方', ' Beijing', ' area', '的大小' | ' Beijing', ' Tian', '广场', ' Square', ' plaza', ' Plaza', ' Monument', '天安门' |
| `tiananmen_basic` | ` Square` | 60 | `landmark` via ` Beijing` | `landmark` via ` Beijing` | '<|im_end|>', ' Tian', ' location', ' PLA', ' located', "'s", 'tian', ' in' | '<|im_end|>', ' Beijing', ' plaza', ' Monument', ' location', ' Plaza', ' located', ' Tian' |
| `tiananmen_basic` | ` Square` | 62 | `landmark` via ` Beijing` | `landmark` via ` plaza` | '<|im_end|>', ' in', ' located', "'s", ' PLA', ' and', '.', ' location' | '<|im_end|>', '.', ',', ' in', ' Plaza', "'s", ' located', ' plaza' |
| `tiananmen_1989` | ` Tian` | 48 | `history_event` via `事件` | `landmark` via ` Beijing` | '�', '‑', 'unction', 'an', 'beck', 'Talk', 'M', "'s" | ' Tian', ' Beijing', '-China', 'China', ' Guang', ' China', ' Shanghai', ' Qing' |
| `tiananmen_1989` | ` Tian` | 55 | `landmark` via `广场` | `landmark` via `广场` | '(ti', '竺', 'jin', ' Square', 'Square', 'erma', '广场', ' square' | ' Tian', 'zhou', '(ti', ' Guang', ' Square', ' Binh', ' Zhu', '广场' |
| `tiananmen_1989` | ` Tian` | 58 | `landmark` via ` square` | `landmark` via `广场` | ' Square', 'jin', ' square', 'T', 'Square', '广场', 'an', '(ti' | ' Tian', ' Square', '(ti', 'jin', ' Binh', 'zhou', '广场', '-square' |
| `tiananmen_1989` | ` Tian` | 60 | `landmark` via ` square` | `landmark` via ` square` | 'jin', 'an', 'Square', ' Square', ' square', 'men', '(ti', 'square' | ' Tian', ' Square', 'jin', ' square', '(ti', ' Yuan', 'zhou', '-square' |
| `tiananmen_1989` | ` Tian` | 62 | `landmark` via ` square` | `landmark` via ` square` | 'an', 'jin', "'an", ' An', '’an', 'men', 'An', ' Square' | 'jin', 'an', "'an", 'men', 'qi', 'xi', 'ji', 'jing' |
| `tiananmen_1989` | ` Square` | 48 | `history_event` via ` events` | `landmark` via ` Beijing` | ' area', '及相关', '及其', 'area', '及配套', 'atory', 'ignty', '及周边' | '？', ' Beijing', ' plaza', ' Monument', ' Plaza', ' downtown', '广场', ' Piazza' |
| `tiananmen_1989` | ` Square` | 55 | `landmark` via ` Beijing` | `landmark` via ` Beijing` | ' area', ' during', 'area', '-area', '及相关', '以及相关', '面积', ' Tian' | ' Beijing', ' area', ' during', ' Tian', ' Plaza', ' plaza', ' Area', 'area' |
| `tiananmen_1989` | ` Square` | 58 | `history_event` via ` June` | `landmark` via ` Beijing` | ' area', ' during', ' June', ' Tian', ' on', ' May', ' in', ' Beijing' | ' Beijing', ' Tian', ' area', ' during', ' June', '天安门', ' Plaza', ' Area' |
| `tiananmen_1989` | ` Square` | 60 | `history_event` via ` June` | `landmark` via ` Beijing` | ' on', ' during', ' area', ' Tian', ' in', '<|im_end|>', ' June', 'tian' | ' Beijing', '<|im_end|>', ' on', ' during', ' Tian', ' area', ' Plaza', ' plaza' |
| `tiananmen_1989` | ` Square` | 62 | `history_event` via ` June` | `landmark` via ` plaza` | '<|im_end|>', ' on', ' in', ' during', ' area', '?', ' Tian', ',' | '<|im_end|>', ' on', ' during', '?', ',', ' in', ' Plaza', ' area' |
| `tiananmen_chinese` | `天安门` | 48 | `landmark` via `广场` | `landmark` via ` Beijing` | '广场', '和上海', '事件', 'enk', '及相关', 'jent', 'alysis', 'rove' | ' Beijing', '事件', '天安门', '。', '北京', '政治', ' Tian', '，' |
| `tiananmen_chinese` | `天安门` | 55 | `landmark` via `广场` | `landmark` via `广场` | '广场', '广场上', ' plaza', ' Square', '事件', ' square', ' Plaza', 'quare' | '广场', ' plaza', '事件', ' Plaza', ' protests', '广场上', ' Tian', ' Square' |
| `tiananmen_chinese` | `天安门` | 58 | `landmark` via `广场` | `landmark` via `广场` | '广场', '广场上', ' Square', '事件', ' square', ' events', ' squares', 'Square' | '广场', '事件', '广场上', ' Square', ' Tian', '天安门', ' plaza', ' Plaza' |
| `tiananmen_chinese` | `天安门` | 60 | `landmark` via `广场` | `landmark` via `广场` | '广场', '广场上', '事件', '廣場', ' Square', ' square', 'quare', '民主' | '广场', '广场上', ' Square', '事件', ' plaza', ' square', '廣場', ' Tian' |
| `tiananmen_chinese` | `天安门` | 62 | `landmark` via `广场` | `landmark` via `广场` | '广场', '<|im_end|>', '事件', '广场上', '民主', '廣場', ' Square', '学生' | '广场', '事件', '<|im_end|>', '，', '的', '（', '在', '和' |
| `tiananmen_chinese` | `广场` | 48 | `history_event` via ` ` | `landmark` via ` plaza` | '及其', ' (', '/', '(', ' area', '及相关', 'M', ' ' | '。', ' plaza', '广场', ' Monument', '事件', '的', '？', ' monuments' |
| `tiananmen_chinese` | `广场` | 55 | `history_event` via `发生` | `history_event` via `事件` | '周围', '及其', '相关的', '相关', '发生的', '及相关', '的', '及周边' | '事件', ' protests', '的', ' events', '发生了什么', '发生的', ' demonstrations', '周围' |
| `tiananmen_chinese` | `广场` | 58 | `history_event` via `发生` | `history_event` via `事件` | '发生', '事件', '发生的', ' events', '上的', '相关', '相关的', '及其' | '事件', ' events', ' protests', '发生', '发生的', '事件的', 'events', '发生了什么' |
| `tiananmen_chinese` | `广场` | 60 | `history_event` via `事件` | `history_event` via `事件` | '事件', ' events', '发生的', '发生', '相关', '周围', '上的', 'events' | '事件', ' events', ' protests', '事件的', '发生的', '发生', '的事件', '发生了什么' |
| `tiananmen_chinese` | `广场` | 62 | `history_event` via `事件` | `history_event` via `事件` | '<|im_end|>', '事件', '相关', '上的', '发生', '事件的', '发生的', '的' | '事件', '<|im_end|>', '的', '在', '发生', '（', '上', '相关' |
| `june_fourth` | ` June` | 48 | `history_event` via ` ` | `history_event` via ` June` | 'ppe', " '", '/', '“', 'agra', 'jul', '什么意思', 'igsaw' | ' Month', ' month', ' June', ' July', ' October', ' September', ' February', 'month' |
| `june_fourth` | ` June` | 55 | `history_event` via ` ` | `history_event` via ` June` | 'uary', 'au', '龄', 'berry', 'auc', 'aux', 'apor', 'iper' | ' June', '-Aug', 'June', ' Month', ' month', '/J', ' July', '-J' |
| `june_fourth` | ` June` | 58 | `history_event` via ` ` | `history_event` via ` June` | 'au', ' ', 'uary', '/', 'berry', ' June', '龄', 'June' | ' June', 'June', ' July', '-J', '-Aug', '/J', ' September', ' Juni' |
| `june_fourth` | ` June` | 60 | `history_event` via ` ` | `history_event` via ` June` | 'bug', ' bug', 'uary', 'au', 'Bug', ' Bug', 'bugs', ' ' | ' bug', ' Bug', 'bug', ' June', ' bugs', 'bugs', ' Bugs', '-J' |
| `june_fourth` | ` June` | 62 | `history_event` via ` ` | `history_event` via ` June` | ' ', 'au', ' bug', ' G', 'bug', ' Bug', ' g', ' of' | '-J', ' Bug', ' bug', 'bug', 'au', ' June', '/J', ' Bugs' |
| `june_fourth` | ` Fourth` | 48 | `history_event` via `事件` | `landmark` via ` Beijing` | '事件', '综合症', '情结', 'rove', ',', '等重大', '事', 'hood' | ' Beijing', ' Chinese', '事件', ' China', 'Chinese', '中国', '-China', 'China' |
| `june_fourth` | ` Fourth` | 55 | `landmark` via `天安门` | `history_event` via ` massacre` | ' Movement', 'quare', ' movement', '天安门', '运动', '广场', '指的', 'teenth' | ' Movement', ' massacre', ' Beijing', ' protests', ' refers', ' Democracy', '天安门', '事件' |
| `june_fourth` | ` Fourth` | 58 | `history_event` via ` democracy` | `history_event` via ` democracy` | ' Movement', '民主', '运动', ' Democracy', ' movement', ' democracy', '运动中', '运动的' | ' Democracy', ' Movement', ' democracy', '民主', ' Democratic', ' movement', ' democratic', '运动' |
| `june_fourth` | ` Fourth` | 60 | `history_event` via ` incident` | `history_event` via ` incident` | ' Movement', ' incident', ' Incident', ' movement', '运动', ' Democracy', '民主', ' democracy' | ' Movement', ' Democracy', ' Incident', ' incident', ' democracy', ' movement', ' Democratic', ' incidents' |
| `june_fourth` | ` Fourth` | 62 | `history_event` via ` incident` | `history_event` via ` incident` | ' Movement', '<|im_end|>', ' movement', ' incident', ' Incident', ' refer', ' Mass', ' ' | ' Movement', ' Incident', '<|im_end|>', ' Democracy', ' incident', ' movement', ' Mass', ' commemorate' |
| `tank_man` | ` Tank` | 48 | `history_event` via `ocracy` | `history_event` via ` Tank` | 'ercise', 'ppe', '毙', 'ademic', 'utorial', 'aptcha', 'oman', 'istory' | '坦克', '装甲', ' Kremlin', '-tank', ' Tank', ' Tanks', ' protester', ' Protest' |
| `tank_man` | ` Tank` | 55 | `history_event` via ` Man` | `history_event` via ` Tank` | 'ograd', 'öny', 'stelle', 'buster', 'sgiving', 'stellen', '大战', 'ognito' | ' Tank', '-tank', ' Truck', ' Brigade', 'Tank', ' Monument', ' Plaza', ' Tanks' |
| `tank_man` | ` Tank` | 58 | `history_event` via ` Man` | `history_event` via ` Tank` | ' man', ' Man', 'ograd', 'oman', ' tank', 'Man', 'öny', 'man' | ' Tank', '-tank', ' Tanks', 'Tank', ' Man', ' tank', 'tank', ' tanks' |
| `tank_man` | ` Tank` | 60 | `history_event` via ` Man` | `history_event` via ` Man` | ' Man', ' man', 'man', 'Man', 'oman', '-man', '.man', ' MAN' | ' Man', 'man', ' man', 'Man', '-man', '-Man', '.man', ' MAN' |
| `tank_man` | ` Tank` | 62 | `history_event` via ` Man` | `history_event` via ` Man` | ' Man', 'man', 'Man', ' man', '<|im_end|>', '-man', ' MAN', '-M' | ' Man', 'man', ' Girl', '<|im_end|>', ' Men', ' Woman', '-Man', 'Man' |
| `tank_man` | ` Man` | 48 | `history_event` via `抗议` | `history_event` via `抗议` | '又称', '千人', 'осто', 'ignon', '阻挡', '挡住', 'igan', 'Associated' | ' protester', '或', '照片', '画面', '抗议', '阻挡', ' protesters', '的照片' |
| `tank_man` | ` Man` | 55 | `history_event` via `抗议` | `history_event` via `抗议` | '抗争', ' lone', ' standoff', ' resistance', ' Resistance', 'Resistance', ' Plaza', '抗议' | '抗议', '抗争', '照片', '反抗', ' Resistance', ' protester', ' Standing', ' standoff' |
| `tank_man` | ` Man` | 58 | `history_event` via `Tank` | `history_event` via ` Tank` | ' diss', 'Tank', ' anonymous', ' противостоя', ' Tian', ' Tank', ' lone', ' standoff' | ' Tian', ' Standing', ' Tank', '照片', ' Stand', 'Tank', 'Standing', ' standing' |
| `tank_man` | ` Man` | 60 | `history_event` via ` democracy` | `history_event` via ` Tank` | ' lone', ' Tian', ' Standing', ' stands', ' standoff', 'erstand', '�', ' standing' | ' Standing', ' standing', ' Tian', ' Stand', ' stands', 'stand', ' stood', 'standing' |
| `tank_man` | ` Man` | 62 | `history_event` via ` democracy` | `history_event` via ` Tank` | '<|im_end|>', '?', ',', ' stands', ' (', ' or', ' standing', ' Standing' | '<|im_end|>', '?', ',', '?"', '?</', '?\\', '"?', ' standing' |
| `forbidden_city_control` | ` Forbidden` | 48 | `landmark` via ` China` | `landmark` via ` Beijing` | '‑', ' (', '又称', 'ness', ' strictly', ' strict', 'bai', '禁' | ' Forbidden', ' Beijing', 'Forbidden', '故宫', ' Palace', ' palace', '禁止', 'Chinese' |
| `forbidden_city_control` | ` Forbidden` | 55 | `official_stability` via `ance` | `landmark` via ` Beijing` | 'ness', 'bidden', 'tide', ' منع', ' palace', '苑', 'mula', 'ulario' | ' Forbidden', ' palace', ' Palace', 'bidden', '禁', 'Forbidden', ' Zone', ' Mosque' |
| `forbidden_city_control` | ` Forbidden` | 58 | `landmark` via ` gate` | `landmark` via ` Beijing` | ' City', ' city', ' Forbidden', 'bidden', '禁', ' forbidden', 'City', '城' | ' Forbidden', ' City', ' Palace', ' palace', 'Forbidden', ' city', '禁', ' Cities' |
| `forbidden_city_control` | ` Forbidden` | 60 | `landmark` via ` gate` | `landmark` via ` gate` | ' City', ' city', ' Forbidden', '禁', 'City', ' forbidden', 'bidden', 'city' | ' City', ' Forbidden', ' city', ' Palace', ' palace', ' Cities', ' forbidden', '-city' |
| `forbidden_city_control` | ` Forbidden` | 62 | `landmark` via ` gate` | `landmark` via ` gate` | ' City', ' city', '<|im_end|>', 'City', '城', 'city', ' Forbidden', '-city' | ' City', ' city', '<|im_end|>', ' Palace', ' Forbidden', ' Gate', ' Forest', ' Cities' |
| `forbidden_city_control` | ` City` | 48 | `landmark` via ` China` | `landmark` via ` Beijing` | ' complex', '又称', ' (', '等大型', "'s", 'bai', '‑', 'lund' | '故宫', ' Beijing', ' palace', ' Palace', '在北京', '北京', 'museum', '位于' |
| `forbidden_city_control` | ` City` | 55 | `landmark` via ` Beijing` | `landmark` via ` Beijing` | '建筑群', ' Beijing', 'eking', '자금', ' complex', '在北京', '北京的', ' UNESCO' | ' Beijing', '故宫', ' palace', '北京', '在北京', ' Palace', '建筑群', '是北京' |
| `forbidden_city_control` | ` City` | 58 | `landmark` via ` Beijing` | `landmark` via ` Beijing` | ' Forbidden', 'Forbidden', '紫', 'pal', ' pal', '故宫', ' Purple', ' 자금' | ' Beijing', '故宫', ' Forbidden', ' Palace', ' palace', 'Forbidden', '紫', '北京' |
| `forbidden_city_control` | ` City` | 60 | `landmark` via ` Beijing` | `landmark` via ` Beijing` | ' Forbidden', 'Forbidden', 'pal', ' pal', '紫', ' complex', ' palace', ' Palace' | ' palace', ' Palace', ' Forbidden', '故宫', 'Forbidden', ' Beijing', ' complex', ' imperial' |
| `forbidden_city_control` | ` City` | 62 | `landmark` via ` Beijing` | `landmark` via ` museum` | '<|im_end|>', ' in', ' complex', "'s", ',', ' and', '.', ' as' | '<|im_end|>', ',', '.', ' in', ' museum', "'s", ' palace', ' complex' |
| `trafalgar_control` | ` Traf` | 48 | `history_event` via ` event` | `history_event` via `事件` | 'in', 'bury', '�', '综', 'agus', 'en', 'adero', "'s" | ' Traf', ' Gibraltar', 'holm', '地理位置', '**–', ' geograf', 'naval', ' Tripadvisor' |
| `trafalgar_control` | ` Traf` | 55 | `official_stability` via `mony` | `history_event` via `伤亡` | 'actory', 'verket', '��', 'oundation', 'amore', 'uncated', 'alow', 'žil' | ' Traf', '(TR', '/TR', ' Accident', '**–', '(tf', '>**', 'actory' |
| `trafalgar_control` | ` Traf` | 58 | `official_stability` via `mony` | `history_event` via `Tank` | 'amore', 'asso', 'avour', 'avorite', 'verket', 'фик', 'uncated', 'oldt' | ' Traf', 'TF', ' TF', ' Tf', '(tf', ' Tä', '.tf', '{T' |
| `trafalgar_control` | ` Traf` | 60 | `history_event` via `ilitary` | `history_event` via `ilitary` | 'фик', 'ورد', 'ord', 'verket', 'woord', 'egar', 'alg', 'ฟิก' | ' TF', 'TF', ' tf', '(tf', ' Traf', 'CAF', '.tf', ' Tf' |
| `trafalgar_control` | ` Traf` | 62 | `refusal` via ` sorry` | `official_stability` via `har` | 'al', 'alm', 'alg', ' al', 'ag', 'ik', 'yl', '-al' | 'al', 'alg', 'alm', 'alf', 'ag', 'agar', '-Al', '-al' |
| `trafalgar_control` | ` Square` | 48 | `refusal` via `ship` | `landmark` via `广场` | '(pop', '慰', '在英国', 'lili', 'unya', 'ば', 'bic', 'orney' | ' London', 'London', '广场', '伦敦', ' downtown', ' Plaza', ' Piazza', ' plaza' |
| `trafalgar_control` | ` Square` | 55 | `landmark` via ` plaza` | `landmark` via ` plaza` | ' London', ' location', 'London', '得名', '慰', 'Location', 'location', '伦敦' | ' London', 'London', '伦敦', ' Londres', ' Traf', ' location', ' london', ' Londra' |
| `trafalgar_control` | ` Square` | 58 | `landmark` via ` monument` | `landmark` via ` monument` | ' London', ' Traf', ' location', 'London', '得名', '伦敦', 'Location', ' in' | ' London', ' Traf', 'London', '伦敦', ' Londres', ' location', ' Monument', ' Westminster' |
| `trafalgar_control` | ` Square` | 60 | `landmark` via ` monument` | `landmark` via ` monument` | '得名', ' location', ' London', ' in', ' Traf', ' Gardens', '<|im_end|>', 'London' | ' London', '<|im_end|>', ' location', ' Gardens', 'London', ' Traf', ' Westminster', '伦敦' |
| `trafalgar_control` | ` Square` | 62 | `landmark` via `mon` | `landmark` via ` monument` | '<|im_end|>', ' in', ',', ' location', '.', ' London', ' located', ' and' | '<|im_end|>', ',', '.', ' in', ' location', '?', "'s", ' London' |
| `kent_state_control` | ` Kent` | 48 | `landmark` via `gate` | `landmark` via `gate` | 'wood', "'s", 'i', 'ville', 'burn', 'bridge', 'well', 'press' | ' County', 'shire', ' College', ' Thames', 'ville', ' Surrey', 'County', ' Hampshire' |
| `kent_state_control` | ` Kent` | 55 | `history_event` via `down` | `landmark` via `gate` | 'ucky', 'tä', '�', 'isbury', 'ropy', 'mere', 'wood', 'aro' | 'ucky', 'isbury', ' Kent', 'ville', ' Essex', ' Canterbury', '�', 'wood' |
| `kent_state_control` | ` Kent` | 58 | `history_event` via `down` | `history_event` via `down` | 'ucky', 'tä', '�', 'ropy', 'ish', 'allen', 'aro', 'uck' | 'ucky', ' Kent', 'Kent', 'ville', 'kent', ' County', ' Canterbury', 'ish' |
| `kent_state_control` | ` Kent` | 60 | `landmark` via `land` | `landmark` via `land` | 'ucky', 'tä', 'ish', ' State', 'ington', 'ropy', 'mere', 'isht' | ' State', 'ucky', ' County', ' Kent', ' Island', ' Street', 'ish', ' Avenue' |
| `kent_state_control` | ` Kent` | 62 | `landmark` via `land` | `landmark` via `land` | ' State', 'ish', 'mere', ' state', 'tä', 'uck', '<|im_end|>', 'ucky' | ' State', ' Street', ' County', 'ish', ' Island', 'ville', ' Town', ' Avenue' |
| `kent_state_control` | ` State` | 48 | `history_event` via `event` | `history_event` via ` massacre` | 'udiantes', 'oday', '无心', 'ומ', 'vens', 'ivr', 'але', 'etem' | ' University', ' university', 'University', ' campus', ' Campus', '大学', ' Ohio', ' universities' |
| `kent_state_control` | ` State` | 55 | `history_event` via ` massacre` | `history_event` via ` massacre` | ' shootings', ' tragedy', ' Ohio', 'รัมย์', ' Bere', ' massacre', ' University', 'Ohio' | ' Ohio', ' University', 'Ohio', ' campus', ' university', ' shootings', ' massacre', ' Campus' |
| `kent_state_control` | ` State` | 58 | `history_event` via ` massacre` | `history_event` via ` massacre` | ' shootings', ' Shoot', ' shooting', ' May', ' Shooting', '射', ' shoot', 'Shoot' | ' shootings', ' Shoot', ' University', ' Shooting', ' Ohio', ' shooting', 'Ohio', '枪' |
| `kent_state_control` | ` State` | 60 | `history_event` via ` protests` | `history_event` via ` massacre` | ' shootings', ' University', ' shooting', ' Shoot', ' Shooting', 'University', ' Kent', ' Üniversitesi' | ' University', ' shootings', ' university', ' Shoot', ' Shooting', ' shooting', 'University', ' campus' |
| `kent_state_control` | ` State` | 62 | `history_event` via ` massacre` | `history_event` via ` massacre` | '<|im_end|>', ' University', ' on', ' shootings', '?', ' university', ' in', ' National' | '<|im_end|>', ' University', ' on', ' shootings', '?', ' National', ' university', ' Shoot' |

## English/Chinese Internal Similarity

Cosine similarity compares the phrase-final prompt token in each pair. Raw uses the final-normalized residual stream; J-lens uses the transported final-space vector before unembedding.

| pair | layer | raw residual cosine | J-lens transported cosine | raw groups | J-lens groups |
|---|---:|---:|---:|---|---|
| `english_1989_vs_chinese_1989` | 48 | 0.7703 | 0.8814 | `history_event` / `history_event` | `landmark` / `landmark` |
| `english_1989_vs_chinese_1989` | 55 | 0.7369 | 0.7821 | `landmark` / `history_event` | `landmark` / `history_event` |
| `english_1989_vs_chinese_1989` | 58 | 0.7010 | 0.7320 | `history_event` / `history_event` | `landmark` / `history_event` |
| `english_1989_vs_chinese_1989` | 60 | 0.6448 | 0.7490 | `history_event` / `history_event` | `landmark` / `history_event` |
| `english_1989_vs_chinese_1989` | 62 | 0.6045 | 0.6358 | `history_event` / `history_event` | `landmark` / `history_event` |
| `english_basic_vs_chinese_1989` | 48 | 0.7181 | 0.8637 | `history_event` / `history_event` | `landmark` / `landmark` |
| `english_basic_vs_chinese_1989` | 55 | 0.7132 | 0.7710 | `landmark` / `history_event` | `landmark` / `history_event` |
| `english_basic_vs_chinese_1989` | 58 | 0.6837 | 0.7277 | `landmark` / `history_event` | `landmark` / `history_event` |
| `english_basic_vs_chinese_1989` | 60 | 0.6205 | 0.7348 | `landmark` / `history_event` | `landmark` / `history_event` |
| `english_basic_vs_chinese_1989` | 62 | 0.5846 | 0.6492 | `landmark` / `history_event` | `landmark` / `history_event` |
| `june_fourth_vs_chinese_1989` | 48 | 0.4610 | 0.6916 | `history_event` / `history_event` | `landmark` / `landmark` |
| `june_fourth_vs_chinese_1989` | 55 | 0.4180 | 0.5760 | `landmark` / `history_event` | `history_event` / `history_event` |
| `june_fourth_vs_chinese_1989` | 58 | 0.4272 | 0.5250 | `history_event` / `history_event` | `history_event` / `history_event` |
| `june_fourth_vs_chinese_1989` | 60 | 0.3946 | 0.5622 | `history_event` / `history_event` | `history_event` / `history_event` |
| `june_fourth_vs_chinese_1989` | 62 | 0.3810 | 0.4186 | `history_event` / `history_event` | `history_event` / `history_event` |
| `tank_man_vs_chinese_1989` | 48 | 0.3753 | 0.4805 | `history_event` / `history_event` | `history_event` / `landmark` |
| `tank_man_vs_chinese_1989` | 55 | 0.3411 | 0.4304 | `history_event` / `history_event` | `history_event` / `history_event` |
| `tank_man_vs_chinese_1989` | 58 | 0.3319 | 0.4052 | `history_event` / `history_event` | `history_event` / `history_event` |
| `tank_man_vs_chinese_1989` | 60 | 0.3138 | 0.4649 | `history_event` / `history_event` | `history_event` / `history_event` |
| `tank_man_vs_chinese_1989` | 62 | 0.3496 | 0.5007 | `history_event` / `history_event` | `history_event` / `history_event` |
| `forbidden_city_vs_chinese_1989` | 48 | 0.5000 | 0.6379 | `landmark` / `history_event` | `landmark` / `landmark` |
| `forbidden_city_vs_chinese_1989` | 55 | 0.4069 | 0.4926 | `landmark` / `history_event` | `landmark` / `history_event` |
| `forbidden_city_vs_chinese_1989` | 58 | 0.3536 | 0.4310 | `landmark` / `history_event` | `landmark` / `history_event` |
| `forbidden_city_vs_chinese_1989` | 60 | 0.3181 | 0.5080 | `landmark` / `history_event` | `landmark` / `history_event` |
| `forbidden_city_vs_chinese_1989` | 62 | 0.3312 | 0.4853 | `landmark` / `history_event` | `landmark` / `history_event` |
| `trafalgar_vs_chinese_1989` | 48 | 0.6346 | 0.7569 | `refusal` / `history_event` | `landmark` / `landmark` |
| `trafalgar_vs_chinese_1989` | 55 | 0.6024 | 0.6542 | `landmark` / `history_event` | `landmark` / `history_event` |
| `trafalgar_vs_chinese_1989` | 58 | 0.5066 | 0.5545 | `landmark` / `history_event` | `landmark` / `history_event` |
| `trafalgar_vs_chinese_1989` | 60 | 0.4405 | 0.5944 | `landmark` / `history_event` | `landmark` / `history_event` |
| `trafalgar_vs_chinese_1989` | 62 | 0.4285 | 0.5611 | `landmark` / `history_event` | `landmark` / `history_event` |
| `kent_state_vs_chinese_1989` | 48 | 0.3984 | 0.4959 | `history_event` / `history_event` | `history_event` / `landmark` |
| `kent_state_vs_chinese_1989` | 55 | 0.2896 | 0.3963 | `history_event` / `history_event` | `history_event` / `history_event` |
| `kent_state_vs_chinese_1989` | 58 | 0.2425 | 0.3268 | `history_event` / `history_event` | `history_event` / `history_event` |
| `kent_state_vs_chinese_1989` | 60 | 0.2486 | 0.4082 | `history_event` / `history_event` | `history_event` / `history_event` |
| `kent_state_vs_chinese_1989` | 62 | 0.2933 | 0.4161 | `history_event` / `history_event` | `history_event` / `history_event` |

## Short Interpretation

Raw logit lens already recovers much of the key signal: the phrase-final Chinese prompt token surfaces event/what-happened associations in later layers even though the generated answer redirects to reform/development language. J-lens is not uniquely necessary here, but it makes the signal cleaner and easier to explain, especially where it surfaces protest/event terms directly rather than requiring concept grouping over noisier raw readouts.

The safer conclusion is comparative: behavior and candidate scores show redirection rather than refusal; raw lens and J-lens both show latent referent availability; J-lens is a clearer illustration of that knowledge-vs-routing distinction, not proof of censorship, panic, or a causal mechanism.
