(() => {
  const format = document.querySelector('#format');
  const palette = document.querySelector('#palette');
  const picker = document.querySelector('#picker');
  const input = document.querySelector('#hex-input');
  const help = document.querySelector('#hex-help');
  const preview = document.querySelector('#preview');
  const result = document.querySelector('#custom-result');
  const label = document.querySelector('#custom-format');
  const colors = TURTLE_COLORS.map(([name, hex]) => ({ name, hex }));
  let selected = picker.value.toUpperCase();
  const rgb = hex => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
  const display = (hex, mode, name) => {
    const values = rgb(hex);
    switch (mode) {
      case 'name': return name || colors.filter(color => color.hex === hex).map(color => color.name).join(' / ') || '无匹配名称';
      case 'rgb': return `(${values.join(', ')})`;
      case 'hex': return hex;
      case 'float': return `(${values.map(value => (value / 255).toFixed(3)).join(', ')})`;
      case 'percent': return `(${values.map(value => (value / 255 * 100).toFixed(1) + '%').join(', ')})`;
      case 'cmyk': {
        const channels = values.map(value => value / 255);
        const k = 1 - Math.max(...channels);
        const cmy = k === 1 ? [0, 0, 0] : channels.map(value => (1 - value - k) / (1 - k));
        return `(${[...cmy, k].map(value => (value * 100).toFixed(1) + '%').join(', ')})`;
      }
    }
  };
  const items = colors.map(color => {
    const row = document.createElement('div'); row.className = 'swatch';
    const chip = document.createElement('span'); chip.className = 'swatch-color'; chip.style.backgroundColor = color.hex; chip.setAttribute('aria-hidden', 'true');
    const text = document.createElement('span'); text.className = 'swatch-value';
    row.title = `${color.name} · ${color.hex}`;
    row.append(chip, text); palette.append(row);
    return { color, text };
  });
  function render() {
    items.forEach(({ color, text }) => { text.textContent = display(color.hex, format.value, color.name); });
    label.textContent = format.options[format.selectedIndex].text;
    result.textContent = display(selected, format.value);
    preview.style.backgroundColor = selected;
  }
  function setColor(value) {
    selected = value.toUpperCase(); picker.value = selected; input.value = selected;
    help.textContent = '输入 6 位十六进制色值，如 #9B87F5。'; help.classList.remove('error'); input.removeAttribute('aria-invalid'); render();
  }
  function apply() {
    let value = input.value.trim(); if (/^[0-9a-f]{6}$/i.test(value)) value = '#' + value;
    if (!/^#[0-9a-f]{6}$/i.test(value)) { help.textContent = '请输入有效的 6 位十六进制色值。'; help.classList.add('error'); input.setAttribute('aria-invalid', 'true'); return; }
    setColor(value);
  }
  format.addEventListener('change', render);
  picker.addEventListener('input', () => setColor(picker.value));
  document.querySelector('#apply').addEventListener('click', apply);
  input.addEventListener('keydown', event => { if (event.key === 'Enter') apply(); });
  render();
})();
