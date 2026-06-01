import indexHtml from './index.html';

export function renderHtml(cookies, rows) {
    // 파일로 불러온 HTML 전체 텍스트를 그대로 반환합니다.
    // 추후 서버 데이터를 HTML에 반영해야 한다면 indexHtml.replace('치환할문자열', 데이터) 방식을 사용할 수 있습니다.
    return indexHtml;
}