const API = window.API_URL || 'http://localhost:8005';

let QUESTIONS = [];

async function loadMeta() {
  const r = await fetch(API + '/api/info');
  const info = await r.json();
  document.getElementById('corpus-info').textContent =
    'Чат «' + info.chat_name + '» · ' + info.message_count + ' сообщений';

  const sel = document.getElementById('chat');
  sel.innerHTML = '';
  (info.available_chats || []).forEach(f => {
    const o = document.createElement('option');
    o.value = f; o.textContent = f;
    if (f === info.chat_file) o.selected = true;
    sel.appendChild(o);
  });
  sel.onchange = () => switchChat(sel.value);

  QUESTIONS = info.questions;
  document.getElementById('examples').innerHTML = '';
  const box = document.getElementById('examples');
  for (const q of QUESTIONS) {
    const el = document.createElement('button');
    el.type = 'button';
    el.className = 'chip';
    el.textContent = q.length > 60 ? q.slice(0, 57) + '…' : q;
    el.title = q;
    el.onclick = () => { document.getElementById('q').value = q; doSearch(); };
    box.appendChild(el);
  }
}

async function doSearch() {
  const text = document.getElementById('q').value.trim();
  if (!text) return;
  const btn = document.getElementById('btn');
  const meta = document.getElementById('meta');
  const results = document.getElementById('results');
  const answerBox = document.getElementById('answer');
  btn.disabled = true;
  meta.textContent = 'Ищу…';
  results.innerHTML = '';
  answerBox.innerHTML = '';

  try {
    const r = await fetch(API + '/api/search', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    const data = await r.json();
    if (data.error) {
      meta.innerHTML = '<span class="err">Ошибка: ' + data.error + '</span>';
      return;
    }

    meta.textContent = 'Найдено ' + data.messages.length + ' сообщений за ' + data.elapsed_ms + ' мс';

    if (!data.messages.length) {
      results.innerHTML = '<div class="empty">Ничего не найдено.</div>';
      return;
    }

    data.messages.forEach((m, i) => {
      const div = document.createElement('div');
      div.className = 'msg';

      const head = document.createElement('div');
      head.className = 'msg-head';
      const rank = document.createElement('span');
      rank.className = 'rank';
      rank.textContent = '#' + (i + 1);
      head.appendChild(rank);
      const who = document.createElement('span');
      who.textContent = m.sender + ' · ' + m.date;
      head.appendChild(who);
      if (m.is_forward) {
        const t = document.createElement('span'); t.className = 'tag';
        t.textContent = 'переслано'; head.appendChild(t);
      }
      if (m.is_quote) {
        const t = document.createElement('span'); t.className = 'tag';
        t.textContent = 'с цитатой'; head.appendChild(t);
      }
      div.appendChild(head);

      const body = document.createElement('div');
      body.className = 'msg-text';
      body.textContent = m.text || '(без текста)';
      div.appendChild(body);

      results.appendChild(div);
    });

    if (data.llm_available) {
      generateAnswer(text, data.messages.map(m => m.id));
    }
  } catch (e) {
    meta.innerHTML = '<span class="err">Ошибка запроса: ' + e.message + '</span>';
  } finally {
    btn.disabled = false;
  }
}

async function generateAnswer(text, ids) {
  const box = document.getElementById('answer');
  box.innerHTML = '<div class="answer"><div class="answer-head">Ответ модели</div>' +
                  '<div class="answer-text pulse">генерирую…</div></div>';
  try {
    const r = await fetch(API + '/api/answer', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text, message_ids: ids})
    });
    const d = await r.json();
    if (d.error) { box.innerHTML = ''; return; }

    const wrap = document.createElement('div');
    wrap.className = 'answer';

    const head = document.createElement('div');
    head.className = 'answer-head';
    const badge = document.createElement('span');
    badge.className = 'answer-badge';
    badge.textContent = d.model || 'LLM';
    head.appendChild(badge);
    const t = document.createElement('span');
    t.textContent = 'ответ сгенерирован по ' + (d.sources || []).length +
                    ' верхним сообщениям · ' + d.elapsed_ms + ' мс';
    head.appendChild(t);
    wrap.appendChild(head);

    const body = document.createElement('div');
    body.className = 'answer-text';
    body.textContent = d.answer;
    wrap.appendChild(body);

    if (d.sources && d.sources.length) {
      const src = document.createElement('div');
      src.className = 'answer-sources';
      const label = document.createElement('b');
      label.textContent = 'Источники:';
      src.appendChild(label);
      d.sources.forEach(s => {
        const line = document.createElement('div');
        line.className = 'src';
        const snippet = s.text.replace(/\s+/g, ' ').slice(0, 90);
        line.textContent = '[' + s.n + '] ' + s.sender + ' · ' + s.date + ' — ' + snippet + '…';
        src.appendChild(line);
      });
      wrap.appendChild(src);
    }

    box.innerHTML = '';
    box.appendChild(wrap);
  } catch (e) {
    box.innerHTML = '';
  }
}

document.getElementById('form').addEventListener('submit', e => {
  e.preventDefault();
  doSearch();
});

loadMeta();

async function switchChat(file) {
  const sel = document.getElementById('chat');
  const hint = document.getElementById('chat-hint');
  sel.disabled = true;
  hint.textContent = 'переиндексирую…';
  document.getElementById('results').innerHTML = '';
  document.getElementById('answer').innerHTML = '';
  document.getElementById('meta').textContent = '';
  try {
    const r = await fetch(API + '/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({file})
    });
    const d = await r.json();
    if (!r.ok) { hint.textContent = 'ошибка: ' + (d.detail || r.status); return; }
    hint.textContent = d.chunks + ' чанков проиндексировано';
    await loadMeta();
  } catch (e) {
    hint.textContent = 'ошибка: ' + e.message;
  } finally {
    sel.disabled = false;
  }
}


document.getElementById('upload').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const hint = document.getElementById('chat-hint');
  hint.textContent = 'загружаю ' + file.name + '…';
  const form = new FormData();
  form.append('file', file);
  try {
    const r = await fetch(API + '/api/upload', {method: 'POST', body: form});
    const d = await r.json();
    if (!r.ok) { hint.textContent = 'ошибка: ' + (d.detail || r.status); return; }
    hint.textContent = 'загружено: ' + d.message_count + ' сообщений';
    await switchChat(d.chat_file);
  } catch (err) {
    hint.textContent = 'ошибка: ' + err.message;
  } finally {
    e.target.value = '';
  }
});
